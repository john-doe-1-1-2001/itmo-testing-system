"""testbuild — a compact declarative black-box testing toolkit.

A :class:`Tester` describes a suite.  The executable is deliberately supplied
only to :meth:`Tester.run`, so one suite can be run against any student build.
The built-in pretest checks process mechanics only; task-specific assertions
belong to user-defined :class:`Expected` implementations.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
	Any,
	Callable,
	Dict,
	Iterable,
	List,
	Mapping,
	MutableMapping,
	Optional,
	Protocol,
	Sequence,
	Set,
	TextIO,
	Tuple,
	Union,
)

__version__ = "0.0.5"

PathLike = Union[str, os.PathLike[str]]
Payload = Union[str, bytes]
CheckReturn = Union[None, bool, str, "Verdict"]


class PolicyValue(Enum):
	"""Convenient DSL values shared by return-code and stderr properties."""

	ANY = "any"
	EMPTY = "empty"
	NONEMPTY = "non-empty"
	NONZERO = "non-zero"


ANY = PolicyValue.ANY
EMPTY = PolicyValue.EMPTY
NONEMPTY = PolicyValue.NONEMPTY
NONZERO = PolicyValue.NONZERO


class ReturnCodePolicy(Enum):
	"""How the child's return code is interpreted by the pretest."""

	MATCH_IF_PRESENTED = "match expected return code"
	SHOULD_BE_ZERO = "return code should be zero"
	SHOULD_NOT_BE_ZERO = "return code should not be zero"
	ANY = "do not check return code"

	# Familiar spellings from the previous testers.
	MatchIfPresented = MATCH_IF_PRESENTED
	ShouldBeZero = SHOULD_BE_ZERO
	ShouldNotBeZero = SHOULD_NOT_BE_ZERO


class StderrPolicy(Enum):
	"""The only built-in stderr check is its emptiness."""

	AUTO = "infer from return-code policy"
	EMPTY = "stderr should be empty"
	NONEMPTY = "stderr should not be empty"
	ANY = "do not check stderr"


class DynamicWrapper(Enum):
	NONE = "none"
	VALGRIND = "valgrind"
	GDB = "gdb"
	DR_MEMORY = "drmemory"
	WINDBG = "windbg"

	NO_WRAPPER = NONE
	VALGRIND_ANALYZER = VALGRIND
	GDB_DEBUGGER = GDB
	DR_MEMORY_ANALYZER = DR_MEMORY
	DRMEMORY = DR_MEMORY
	WINDBG_DEBUGGER = WINDBG


class VerdictCode(Enum):
	SUCCESS = "success"
	SKIPPED = "skipped"
	TIMEOUT = "timeout expired"
	LAUNCH_ERROR = "program could not be started"
	RETURN_CODE = "program returned a wrong return code"
	STDERR_EMPTY = "standard error output is empty"
	STDERR_NOT_EMPTY = "standard error output is not empty"
	EXPECTATION = "expectation failed"
	HOOK_ERROR = "hook failed"
	VALGRIND_ERROR = "valgrind error"
	GDB_ERROR = "gdb error"
	DR_MEMORY_ERROR = "Dr. Memory error"
	WINDBG_ERROR = "WinDbg error"
	INTERNAL_ERROR = "internal error"


@dataclass(frozen=True)
class Verdict:
	code: VerdictCode = VerdictCode.SUCCESS
	message: str = ""
	details: Tuple[str, ...] = ()

	def is_success(self) -> bool:
		return self.code is VerdictCode.SUCCESS

	def is_skipped(self) -> bool:
		return self.code is VerdictCode.SKIPPED

	def is_failed(self) -> bool:
		return not self.is_success() and not self.is_skipped()

	def __bool__(self) -> bool:
		return self.is_success()


OK = Verdict()


def skip(message: str = "") -> Verdict:
	return Verdict(VerdictCode.SKIPPED, message)


@dataclass(frozen=True)
class FileSource:
	path: Path
	binary: bool = False

	def read(self, encoding: str) -> Payload:
		if self.binary:
			return self.path.read_bytes()
		return self.path.read_text(encoding=encoding)


def from_file(path: PathLike, *, binary: bool = False) -> FileSource:
	"""Return a lazily-read stdin value backed by ``path``."""

	return FileSource(Path(path).expanduser().resolve(), binary)


InputValue = Union[None, Payload, FileSource]


class Expected(ABC):
	"""Task-specific result checker.

	The runner has already checked timeout, dynamic-analyzer output, return
	code and stderr policy before this method is called.
	"""

	@abstractmethod
	def test(self, run: "Run", runned: "Runned") -> CheckReturn:
		raise NotImplementedError


@dataclass
class Run:
	name: str = "run"
	args: List[str] = field(default_factory=list)
	stdin: Optional[str] = None
	timeout: float = 1.0
	returncode_policy: ReturnCodePolicy = ReturnCodePolicy.SHOULD_BE_ZERO
	expected_returncode: Optional[int] = None
	stderr_policy: StderrPolicy = StderrPolicy.AUTO
	expected: Optional[Expected] = None
	env: Dict[str, str] = field(default_factory=dict)
	skip_wrappers: Set[DynamicWrapper] = field(default_factory=set)


def _copy_run(source: Run, *, name: Optional[str] = None) -> Run:
	"""Copy mutable run settings while intentionally sharing Expected."""

	return Run(
		name=source.name if name is None else name,
		args=list(source.args),
		stdin=source.stdin,
		timeout=source.timeout,
		returncode_policy=source.returncode_policy,
		expected_returncode=source.expected_returncode,
		stderr_policy=source.stderr_policy,
		expected=source.expected,
		env=dict(source.env),
		skip_wrappers=set(source.skip_wrappers),
	)


@dataclass
class Runned:
	executable: Path
	group: str
	test_name: str
	run_name: str
	command: Tuple[str, ...]
	workdir: Path
	returncode: Optional[int]
	stdout: Union[bytes, str]
	stderr: Union[bytes, str]
	duration: float
	dynamic_wrapper: DynamicWrapper
	wrapper_log: Optional[str] = None
	timed_out: bool = False
	launch_error: Optional[str] = None
	encoding: str = "utf-8"

	@property
	def stdout_text(self) -> str:
		if isinstance(self.stdout, str):
			return self.stdout
		return self.stdout.decode(self.encoding, errors="replace")

	@property
	def stderr_text(self) -> str:
		if isinstance(self.stderr, str):
			return self.stderr
		return self.stderr.decode(self.encoding, errors="replace")

	# Compatibility-minded accessors make porting old Expected classes easy.
	def get_returncode(self) -> Optional[int]:
		return self.returncode

	def get_stdout(self) -> str:
		return self.stdout_text

	def get_stderr(self) -> str:
		return self.stderr_text

	def get_workdir(self) -> Path:
		return self.workdir


class Action(Protocol):
	def execute(self, context: "ActionContext") -> None:
		...


@dataclass
class GroupSpec:
	name: str
	categories: Tuple[str, ...] = ()
	tests: List["TestSpec"] = field(default_factory=list)
	defaults: Run = field(default_factory=Run)


@dataclass
class TestSpec:
	index: int
	name: str
	categories: Tuple[str, ...] = ()
	run_defaults: Run = field(default_factory=Run)
	runs: List[Run] = field(default_factory=list)
	before: List[Action] = field(default_factory=list)
	after: List[Action] = field(default_factory=list)
	seed_directory: Optional[Path] = None


@dataclass
class ActionContext:
	executable: Path
	dynamic_wrapper: DynamicWrapper
	group: GroupSpec
	test: TestSpec
	workdir: Path
	env: MutableMapping[str, str]
	state: Dict[str, Any] = field(default_factory=dict)
	runned: Optional[Runned] = None

	def path(self, path: PathLike) -> Path:
		candidate = Path(path)
		return candidate if candidate.is_absolute() else self.workdir / candidate


def _inside(context: ActionContext, path: PathLike) -> Path:
	return context.path(path)


@dataclass
class RemoveFile:
	path: PathLike

	def execute(self, context: ActionContext) -> None:
		_inside(context, self.path).unlink(missing_ok=True)


@dataclass
class RemoveTree:
	path: PathLike

	def execute(self, context: ActionContext) -> None:
		path = _inside(context, self.path)
		if path.exists():
			shutil.rmtree(path)


@dataclass
class CreateFile:
	path: PathLike
	contents: Payload = ""
	encoding: str = "utf-8"

	def execute(self, context: ActionContext) -> None:
		path = _inside(context, self.path)
		path.parent.mkdir(parents=True, exist_ok=True)
		if isinstance(self.contents, bytes):
			path.write_bytes(self.contents)
		else:
			path.write_text(self.contents, encoding=self.encoding)


@dataclass
class MakeDirectory:
	path: PathLike
	parents: bool = True

	def execute(self, context: ActionContext) -> None:
		_inside(context, self.path).mkdir(parents=self.parents, exist_ok=True)


@dataclass
class CopyFile:
	source: Path
	destination: PathLike

	def execute(self, context: ActionContext) -> None:
		destination = _inside(context, self.destination)
		destination.parent.mkdir(parents=True, exist_ok=True)
		shutil.copy2(self.source, destination)


@dataclass
class SetEnvironment:
	name: str
	value: Optional[str]

	def execute(self, context: ActionContext) -> None:
		if self.value is None:
			context.env.pop(self.name, None)
		else:
			context.env[self.name] = self.value


@dataclass
class RunnableAction:
	runnable: Callable[[], Any]

	def execute(self, context: ActionContext) -> None:
		self.runnable()


@dataclass
class ContextRunnableAction:
	runnable: Callable[[ActionContext], Any]

	def execute(self, context: ActionContext) -> None:
		self.runnable(context)


class HookBuilder:
	def __init__(self, actions: List[Action]) -> None:
		self.__actions = actions

	def __enter__(self) -> "HookBuilder":
		return self

	def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
		return False

	def remove_file(self, path: PathLike) -> None:
		self.__actions.append(RemoveFile(path))

	def remove_tree(self, path: PathLike) -> None:
		self.__actions.append(RemoveTree(path))

	def create_file(
		self, path: PathLike, contents: Payload = "", *, encoding: str = "utf-8"
	) -> None:
		self.__actions.append(CreateFile(path, contents, encoding))

	write_file = create_file

	def mkdir(self, path: PathLike, *, parents: bool = True) -> None:
		self.__actions.append(MakeDirectory(path, parents))

	def copy_file(self, source: PathLike, destination: PathLike) -> None:
		self.__actions.append(
			CopyFile(Path(source).expanduser().resolve(), destination)
		)

	def set_env(self, name: str, value: str) -> None:
		self.__actions.append(SetEnvironment(name, value))

	def unset_env(self, name: str) -> None:
		self.__actions.append(SetEnvironment(name, None))

	def do(self, runnable: Callable[[], Any]) -> None:
		"""Schedule a zero-argument Runnable-like callable."""

		self.__actions.append(RunnableAction(runnable))

	def do_with_context(self, runnable: Callable[[ActionContext], Any]) -> None:
		"""Schedule a callable receiving only the current ActionContext."""

		self.__actions.append(ContextRunnableAction(runnable))


class RunBuilder:
	def __init__(self, spec: Run) -> None:
		self.__spec = spec

	def __enter__(self) -> "RunBuilder":
		return self

	def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
		return False

	@property
	def args(self) -> List[str]:
		return self.__spec.args

	@args.setter
	def args(self, value: Iterable[Any]) -> None:
		self.__spec.args = [str(item) for item in value]

	@property
	def stdin(self) -> Optional[str]:
		return self.__spec.stdin

	@stdin.setter
	def stdin(self, value: Optional[str]) -> None:
		self.__spec.stdin = value

	@property
	def timeout(self) -> float:
		return self.__spec.timeout

	@timeout.setter
	def timeout(self, value: Union[int, float]) -> None:
		if float(value) <= 0:
			raise ValueError("timeout must be greater than zero")
		self.__spec.timeout = float(value)

	@property
	def env(self) -> Dict[str, str]:
		return self.__spec.env

	@env.setter
	def env(self, value: Mapping[str, Any]) -> None:
		self.__spec.env = {str(key): str(item) for key, item in value.items()}

	@property
	def returncode_policy(self) -> ReturnCodePolicy:
		return self.__spec.returncode_policy

	@returncode_policy.setter
	def returncode_policy(self, value: ReturnCodePolicy) -> None:
		self.__spec.returncode_policy = ReturnCodePolicy(value)

	@property
	def returncode(self) -> Union[int, PolicyValue, None]:
		policy = self.__spec.returncode_policy
		if policy is ReturnCodePolicy.SHOULD_NOT_BE_ZERO:
			return NONZERO
		if policy is ReturnCodePolicy.ANY:
			return ANY
		return self.__spec.expected_returncode

	@returncode.setter
	def returncode(self, value: Union[int, PolicyValue]) -> None:
		if value is NONZERO:
			self.__spec.returncode_policy = ReturnCodePolicy.SHOULD_NOT_BE_ZERO
			self.__spec.expected_returncode = None
		elif value is ANY:
			self.__spec.returncode_policy = ReturnCodePolicy.ANY
			self.__spec.expected_returncode = None
		elif value is EMPTY or value is NONEMPTY:
			raise ValueError("EMPTY/NONEMPTY are stderr policies, not return codes")
		else:
			self.__spec.returncode_policy = ReturnCodePolicy.MATCH_IF_PRESENTED
			self.__spec.expected_returncode = int(value)

	@property
	def expected_returncode(self) -> Optional[int]:
		return self.__spec.expected_returncode

	@expected_returncode.setter
	def expected_returncode(self, value: Optional[int]) -> None:
		self.__spec.expected_returncode = None if value is None else int(value)

	@property
	def stderr_policy(self) -> StderrPolicy:
		return self.__spec.stderr_policy

	@stderr_policy.setter
	def stderr_policy(self, value: StderrPolicy) -> None:
		self.__spec.stderr_policy = StderrPolicy(value)

	@property
	def stderr(self) -> PolicyValue:
		policy = self.__spec.stderr_policy
		return {
			StderrPolicy.EMPTY: EMPTY,
			StderrPolicy.NONEMPTY: NONEMPTY,
			StderrPolicy.ANY: ANY,
			StderrPolicy.AUTO: ANY,
		}[policy]

	@stderr.setter
	def stderr(self, value: PolicyValue) -> None:
		if value is EMPTY:
			self.__spec.stderr_policy = StderrPolicy.EMPTY
		elif value is NONEMPTY:
			self.__spec.stderr_policy = StderrPolicy.NONEMPTY
		elif value is ANY:
			self.__spec.stderr_policy = StderrPolicy.ANY
		else:
			raise TypeError(
				"stderr accepts only testbuild.EMPTY, NONEMPTY, or ANY"
			)

	@property
	def expected(self) -> Optional[Expected]:
		return self.__spec.expected

	@expected.setter
	def expected(self, value: Optional[Expected]) -> None:
		if value is not None and not isinstance(value, Expected):
			raise TypeError("expected must be an instance of testbuild.Expected")
		self.__spec.expected = value

	def skip_when(self, *wrappers: DynamicWrapper) -> None:
		self.__spec.skip_wrappers.update(DynamicWrapper(item) for item in wrappers)


class TestBuilder(RunBuilder):
	def __init__(self, spec: TestSpec) -> None:
		self.__test_spec = spec
		self.__explicit_runs = False
		super().__init__(self.__default_run())

	def __enter__(self) -> "TestBuilder":
		return self

	def __default_run(self) -> Run:
		if not self.__test_spec.runs:
			self.__test_spec.runs.append(_copy_run(self.__test_spec.run_defaults))
		return self.__test_spec.runs[0]

	@property
	def categories(self) -> Tuple[str, ...]:
		return self.__test_spec.categories

	@categories.setter
	def categories(self, value: Iterable[str]) -> None:
		self.__test_spec.categories = tuple(dict.fromkeys(map(str, value)))

	def step(self, name: Optional[str] = None) -> RunBuilder:
		implicit = self.__test_spec.runs[0]
		if not self.__explicit_runs:
			self.__explicit_runs = True
			implicit.name = name or "run 1"
			return RunBuilder(implicit)
		spec = _copy_run(
			self.__test_spec.run_defaults,
			name=name or f"run {len(self.__test_spec.runs) + 1}",
		)
		self.__test_spec.runs.append(spec)
		return RunBuilder(spec)

	def run(self, name: Optional[str] = None) -> RunBuilder:
		return self.step(name)

	def before_test(self) -> HookBuilder:
		return HookBuilder(self.__test_spec.before)

	def after_test(self) -> HookBuilder:
		return HookBuilder(self.__test_spec.after)


class MultiRunTestBuilder(TestBuilder):
	def __enter__(self) -> "MultiRunTestBuilder":
		return self


class GroupBuilder(RunBuilder):
	def __init__(self, spec: GroupSpec) -> None:
		self.__spec = spec
		super().__init__(spec.defaults)

	def __enter__(self) -> "GroupBuilder":
		return self

	def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
		return False

	def test(
		self,
		name: str,
		*,
		categories: Iterable[str] = (),
		seed_directory: Optional[PathLike] = None,
	) -> TestBuilder:
		return TestBuilder(self.__make_test(name, categories, seed_directory))

	def multi_run_test(
		self,
		name: str,
		*,
		categories: Iterable[str] = (),
		seed_directory: Optional[PathLike] = None,
	) -> MultiRunTestBuilder:
		"""Declare several independent launches sharing one workdir."""

		return MultiRunTestBuilder(
			self.__make_test(name, categories, seed_directory)
		)

	def __make_test(
		self,
		name: str,
		categories: Iterable[str],
		seed_directory: Optional[PathLike],
	) -> TestSpec:
		merged = tuple(
			dict.fromkeys([*self.__spec.categories, *(str(item) for item in categories)])
		)
		spec = TestSpec(
			index=len(self.__spec.tests),
			name=name,
			categories=merged,
			run_defaults=_copy_run(self.__spec.defaults),
			seed_directory=(
				None
				if seed_directory is None
				else Path(seed_directory).expanduser().resolve()
			),
		)
		self.__spec.tests.append(spec)
		return spec


@dataclass
class TestResult:
	group: str
	index: int
	name: str
	categories: Tuple[str, ...]
	verdict: Verdict
	runs: List[Runned] = field(default_factory=list)

	@property
	def passed(self) -> bool:
		return self.verdict.is_success()

	@property
	def skipped(self) -> bool:
		return self.verdict.is_skipped()

	def as_dict(self) -> Dict[str, Any]:
		return {
			"group": self.group,
			"id": self.index,
			"name": self.name,
			"categories": list(self.categories),
			"passed": self.passed,
			"skipped": self.skipped,
			"verdict": self.verdict.code.value,
			"message": self.verdict.message,
			"details": list(self.verdict.details),
			"runs": [
				{
					"name": run.run_name,
					"command": list(run.command),
					"returncode": run.returncode,
					"stdout": run.stdout_text,
					"stderr": run.stderr_text,
					"duration_seconds": run.duration,
					"timed_out": run.timed_out,
					"launch_error": run.launch_error,
					"dynamic_wrapper": run.dynamic_wrapper.value,
					"wrapper_log": run.wrapper_log or "",
				}
				for run in self.runs
			],
		}


@dataclass
class Report:
	suite: str
	executable: Path
	dynamic_wrapper: DynamicWrapper
	results: List[TestResult]
	duration: float
	coefficients: Mapping[str, float] = field(default_factory=dict)

	@property
	def passed(self) -> int:
		return sum(result.passed for result in self.results)

	@property
	def skipped(self) -> int:
		return sum(result.skipped for result in self.results)

	@property
	def total(self) -> int:
		return len(self.results) - self.skipped

	@property
	def exit_code(self) -> int:
		return 0 if self.ok() else 1

	def ok(self) -> bool:
		return self.passed == self.total

	def category_scores(self) -> Dict[str, float]:
		categories = set(self.coefficients)
		for result in self.results:
			categories.update(result.categories)
		answer: Dict[str, float] = {}
		for category in sorted(categories):
			relevant = [
				result
				for result in self.results
				if category in result.categories and not result.skipped
			]
			answer[category] = (
				sum(result.passed for result in relevant) / len(relevant)
				if relevant
				else 1.0
			)
		return answer

	@property
	def score(self) -> float:
		scores = self.category_scores()
		return sum(
			scores.get(category, 0.0) * coefficient
			for category, coefficient in self.coefficients.items()
		)

	def as_dict(self) -> Dict[str, Any]:
		return {
			"suite": self.suite,
			"executable": str(self.executable),
			"dynamic_wrapper": self.dynamic_wrapper.value,
			"passed": self.passed,
			"total": self.total,
			"skipped": self.skipped,
			"exitcode": self.exit_code,
			"duration_seconds": self.duration,
			"score": self.score,
			"categories": self.category_scores(),
			"tests": [result.as_dict() for result in self.results],
		}

	def export_json(self, path: PathLike) -> None:
		Path(path).write_text(
			json.dumps(self.as_dict(), indent=2, ensure_ascii=False) + "\n",
			encoding="utf-8",
		)


class Tester:
	"""A reusable test-suite description with no executable bound to it."""

	__test__ = False

	def __init__(
		self,
		name: str,
		*,
		coefficients: Optional[Mapping[str, float]] = None,
		env: Optional[Mapping[str, str]] = None,
		encoding: str = "utf-8",
		temporary_directory: Optional[PathLike] = None,
		keep_workdirs: bool = False,
		output: Optional[TextIO] = None,
	) -> None:
		self.name = name
		self.coefficients = dict(coefficients or {})
		self.env = {str(key): str(value) for key, value in (env or {}).items()}
		self.encoding = encoding
		self.temporary_directory = (
			None
			if temporary_directory is None
			else Path(temporary_directory).expanduser().resolve()
		)
		self.keep_workdirs = keep_workdirs
		self.output = output
		self.__groups: List[GroupSpec] = []

	def group(self, name: str, *, categories: Iterable[str] = ()) -> GroupBuilder:
		spec = GroupSpec(name, tuple(dict.fromkeys(map(str, categories))))
		self.__groups.append(spec)
		return GroupBuilder(spec)

	def run(
		self,
		executable_path: PathLike,
		*,
		timeout_factor: float = 1.0,
		dynamic_wrapper: DynamicWrapper = DynamicWrapper.NONE,
		quiet: bool = False,
	) -> Report:
		"""Run the already-built suite against ``executable_path``."""

		if timeout_factor <= 0:
			raise ValueError("timeout_factor must be greater than zero")
		executable = Path(executable_path).expanduser().resolve()
		if not executable.is_file():
			raise FileNotFoundError(f"executable not found: {executable}")
		wrapper = DynamicWrapper(dynamic_wrapper)
		self.__ensure_wrapper_available(wrapper)
		if self.temporary_directory is not None:
			self.temporary_directory.mkdir(parents=True, exist_ok=True)

		started = time.monotonic()
		results: List[TestResult] = []
		for group in self.__groups:
			for test in group.tests:
				result = self.__run_test(
					executable, wrapper, timeout_factor, group, test
				)
				results.append(result)
				if not quiet:
					self.__print_result(result)
		report = Report(
			self.name,
			executable,
			wrapper,
			results,
			time.monotonic() - started,
			self.coefficients,
		)
		if not quiet:
			print(
				f"{report.passed}/{report.total} tests passed "
				f"({report.skipped} skipped) in {report.duration:.3f}s",
				file=self.output,
			)
		return report

	def warm(
		self,
		executable_path: PathLike,
		*,
		timeout_factor: float = 1.0,
	) -> None:
		self.run(executable_path, timeout_factor=timeout_factor, quiet=True)

	@staticmethod
	def __ensure_wrapper_available(wrapper: DynamicWrapper) -> None:
		if wrapper is DynamicWrapper.NONE:
			return
		if wrapper in (DynamicWrapper.DR_MEMORY, DynamicWrapper.WINDBG):
			if os.name != "nt":
				raise OSError(f"{wrapper.value} is available only on Windows")
		executable = {
			DynamicWrapper.DR_MEMORY: "drmemory.exe",
			DynamicWrapper.WINDBG: "cdb.exe",
		}.get(wrapper, wrapper.value)
		if shutil.which(executable) is None:
			raise FileNotFoundError(
				f"dynamic wrapper is not installed or not in PATH: {executable}"
			)

	def __print_result(self, result: TestResult) -> None:
		marker = "SKIP" if result.skipped else "PASS" if result.passed else "FAIL"
		print(f"[{marker}] {result.group}/{result.name}", file=self.output)
		if result.verdict.is_failed():
			suffix = f": {result.verdict.message}" if result.verdict.message else ""
			print(f"       {result.verdict.code.value}{suffix}", file=self.output)
			for line in result.verdict.details:
				print(f"       {line}", file=self.output)

	def __run_test(
		self,
		executable: Path,
		wrapper: DynamicWrapper,
		timeout_factor: float,
		group: GroupSpec,
		test: TestSpec,
	) -> TestResult:
		safe_suite = re.sub(r"[^A-Za-z0-9_.-]+", "-", self.name)[:32] or "suite"
		workdir = Path(
			tempfile.mkdtemp(
				prefix=f"testbuild-{safe_suite}-{os.getpid()}-",
				dir=self.temporary_directory,
			)
		)
		environment = dict(os.environ)
		environment.update(self.env)
		context = ActionContext(executable, wrapper, group, test, workdir, environment)
		runned_items: List[Runned] = []
		verdict = OK
		try:
			if test.seed_directory is not None:
				shutil.copytree(test.seed_directory, workdir, dirs_exist_ok=True)
			verdict = self.__execute_actions(test.before, context, "before")
			if verdict:
				for run in test.runs:
					if wrapper in run.skip_wrappers:
						verdict = skip(f"{wrapper.value} is disabled for this run")
						break
					runned = self.__execute_run(
						executable,
						wrapper,
						timeout_factor,
						group,
						test,
						run,
						context,
					)
					runned_items.append(runned)
					context.runned = runned
					verdict = self.__pretest(run, runned)
					if verdict and run.expected is not None:
						verdict = self.__invoke_expected(run.expected, run, runned)
					if not verdict:
						break
		except Exception as error:
			verdict = Verdict(
				VerdictCode.INTERNAL_ERROR, f"{type(error).__name__}: {error}"
			)
		finally:
			cleanup = self.__execute_actions(test.after, context, "after")
			if verdict and not cleanup:
				verdict = cleanup
			elif cleanup.is_failed():
				verdict = Verdict(
					verdict.code,
					verdict.message,
					(*verdict.details, f"cleanup: {cleanup.message}"),
				)
			if not self.keep_workdirs:
				shutil.rmtree(workdir, ignore_errors=True)

		return TestResult(
			group.name,
			test.index,
			test.name,
			test.categories,
			verdict,
			runned_items,
		)

	@staticmethod
	def __execute_actions(
		actions: Iterable[Action], context: ActionContext, stage: str
	) -> Verdict:
		try:
			for action in actions:
				action.execute(context)
			return OK
		except Exception as error:
			return Verdict(
				VerdictCode.HOOK_ERROR,
				f"{stage} hook: {type(error).__name__}: {error}",
			)

	def __execute_run(
		self,
		executable: Path,
		wrapper: DynamicWrapper,
		timeout_factor: float,
		group: GroupSpec,
		test: TestSpec,
		run: Run,
		context: ActionContext,
	) -> Runned:
		wrapper_log_path = context.workdir / f".testbuild-{wrapper.value}.log"
		if wrapper is DynamicWrapper.DR_MEMORY:
			wrapper_log_path.mkdir(exist_ok=True)
		command = self.__command(executable, run.args, wrapper, wrapper_log_path)
		environment = dict(context.env)
		environment.update(run.env)
		stdin = run.stdin
		started = time.monotonic()
		process: Optional[subprocess.Popen] = None
		try:
			process = subprocess.Popen(
				command,
				universal_newlines=True,
				cwd=context.workdir,
				env=environment,
				stdin=subprocess.PIPE,
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE,
				start_new_session=(os.name == "posix"),
			)
			try:
				stdout, stderr = process.communicate(
					input=stdin, timeout=run.timeout * timeout_factor
				)
				return self.__make_runned(
					executable,
					group,
					test,
					run,
					command,
					context.workdir,
					process.returncode,
					stdout,
					stderr,
					started,
					wrapper,
					wrapper_log_path,
				)
			except subprocess.TimeoutExpired:
				self.__kill_process(process)
				stdout, stderr = process.communicate()
				return self.__make_runned(
					executable,
					group,
					test,
					run,
					command,
					context.workdir,
					None,
					stdout,
					stderr,
					started,
					wrapper,
					wrapper_log_path,
					timed_out=True,
				)
		except Exception as error:
			if process is not None and process.poll() is None:
				self.__kill_process(process)
				process.communicate()
			return self.__make_runned(
				executable,
				group,
				test,
				run,
				command,
				context.workdir,
				None,
				b"",
				b"",
				started,
				wrapper,
				wrapper_log_path,
				launch_error=f"{type(error).__name__}: {error}",
			)

	@staticmethod
	def __kill_process(process: subprocess.Popen[bytes]) -> None:
		if process.poll() is not None:
			return
		try:
			if os.name == "posix":
				os.killpg(process.pid, signal.SIGKILL)
			elif os.name == "nt":
				subprocess.run(
					["taskkill", "/PID", str(process.pid), "/T", "/F"],
					stdout=subprocess.DEVNULL,
					stderr=subprocess.DEVNULL,
					check=False,
				)
				if process.poll() is None:
					process.kill()
			else:
				process.kill()
		except ProcessLookupError:
			pass

	@staticmethod
	def __command(
		executable: Path,
		args: Sequence[str],
		wrapper: DynamicWrapper,
		log_path: Path,
	) -> List[str]:
		if wrapper is DynamicWrapper.VALGRIND:
			return [
				"valgrind",
				"--tool=memcheck",
				"--leak-check=full",
				"--show-leak-kinds=all",
				"--errors-for-leak-kinds=all",
				"--track-origins=yes",
				"--error-exitcode=125",
				f"--log-file={log_path}",
				str(executable),
				*args,
			]
		if wrapper is DynamicWrapper.GDB:
			return [
				"gdb",
				"-q",
				"--batch-silent",
				"-return-child-result",
				"-ex",
				"set pagination off",
				"-ex",
				"set confirm off",
				"-ex",
				f"set logging file {log_path}",
				"-ex",
				"set logging redirect on",
				"-ex",
				"set logging overwrite on",
				"-ex",
				"set logging enabled on",
				"-ex",
				"run",
				"-ex",
				"thread apply all backtrace full",
				"--args",
				str(executable),
				*args,
			]
		if wrapper is DynamicWrapper.DR_MEMORY:
			# Trial variant: add "-light" for faster but less complete checking.
			# Trial variant: replace "-quiet" with "-results_to_stderr" to
			# stream reports, at the cost of mixing them with application stderr.
			drmemory_options = [
				"-batch",
				"-quiet",
				"-exit_code_if_errors",
				"125",
				"-logdir",
				str(log_path),
			]
			return [
				"drmemory.exe",
				*drmemory_options,
				"--",
				str(executable),
				*args,
			]
		if wrapper is DynamicWrapper.WINDBG:
			# GUI trial variants.  cdb.exe is the WinDbg engine's console
			# frontend and is the predictable choice for unattended tests.
			# debugger = "windbg.exe"  # WinDbg Classic; add "-Q".
			# debugger = "WinDbgX.exe"  # Modern WinDbg; may keep its UI open.
			debugger = "cdb.exe"
			commands = (
				'sxe -c2 ".echo TESTBUILD_FATAL_EXCEPTION; '
				'!analyze -v; q" *; g'
			)
			return [
				debugger,
				"-G",
				"-logo",
				str(log_path),
				"-c",
				commands,
				# Trial variant: add "-o" to debug child processes as well.
				str(executable),
				*args,
			]
		return [str(executable), *args]

	def __make_runned(
		self,
		executable: Path,
		group: GroupSpec,
		test: TestSpec,
		run: Run,
		command: Sequence[str],
		workdir: Path,
		returncode: Optional[int],
		stdout: bytes,
		stderr: bytes,
		started: float,
		wrapper: DynamicWrapper,
		wrapper_log_path: Path,
		*,
		timed_out: bool = False,
		launch_error: Optional[str] = None,
	) -> Runned:
		wrapper_log: Optional[str] = None
		if wrapper is DynamicWrapper.DR_MEMORY and wrapper_log_path.is_dir():
			results = list(wrapper_log_path.glob("DrMemory-*/results.txt"))
			if results:
				latest = max(results, key=lambda path: path.stat().st_mtime_ns)
				wrapper_log = latest.read_text(encoding="utf-8", errors="replace")
		elif wrapper is not DynamicWrapper.NONE and wrapper_log_path.is_file():
			wrapper_log = wrapper_log_path.read_text(
				encoding="utf-8", errors="replace"
			)
		if wrapper is DynamicWrapper.WINDBG and wrapper_log is not None:
			exit_codes = re.findall(
				r"exited with code\s+(-?\d+)\s+\(0x[0-9a-f]+\)",
				wrapper_log,
				flags=re.IGNORECASE,
			)
			if exit_codes:
				returncode = int(exit_codes[-1])
		return Runned(
			executable,
			group.name,
			test.name,
			run.name,
			tuple(command),
			workdir,
			returncode,
			stdout,
			stderr,
			time.monotonic() - started,
			wrapper,
			wrapper_log,
			timed_out,
			launch_error,
			self.encoding,
		)

	def __payload(self, value: InputValue) -> Optional[bytes]:
		if value is None:
			return None
		if isinstance(value, FileSource):
			value = value.read(self.encoding)
		return value if isinstance(value, bytes) else value.encode(self.encoding)

	def __pretest(self, run: Run, runned: Runned) -> Verdict:
		if runned.launch_error is not None:
			return Verdict(VerdictCode.LAUNCH_ERROR, runned.launch_error)
		if runned.timed_out:
			return Verdict(VerdictCode.TIMEOUT, f"limit was {run.timeout:g}s")

		wrapper_verdict = self.__check_dynamic_wrapper(runned)
		if not wrapper_verdict:
			return wrapper_verdict

		code_verdict = self.__check_returncode(run, runned.returncode)
		if not code_verdict:
			return code_verdict

		stderr_policy = self.__resolve_stderr_policy(run)
		if stderr_policy is StderrPolicy.EMPTY and runned.stderr:
			return Verdict(
				VerdictCode.STDERR_NOT_EMPTY,
				self.__display(runned.stderr, runned.encoding),
			)
		if stderr_policy is StderrPolicy.NONEMPTY and not runned.stderr:
			return Verdict(VerdictCode.STDERR_EMPTY)
		return OK

	@staticmethod
	def __resolve_stderr_policy(run: Run) -> StderrPolicy:
		if run.stderr_policy is not StderrPolicy.AUTO:
			return run.stderr_policy
		if run.returncode_policy is ReturnCodePolicy.SHOULD_NOT_BE_ZERO:
			return StderrPolicy.NONEMPTY
		if run.returncode_policy is ReturnCodePolicy.MATCH_IF_PRESENTED:
			if run.expected_returncode is None:
				return StderrPolicy.ANY
			return (
				StderrPolicy.EMPTY
				if run.expected_returncode == 0
				else StderrPolicy.NONEMPTY
			)
		if run.returncode_policy is ReturnCodePolicy.SHOULD_BE_ZERO:
			return StderrPolicy.EMPTY
		return StderrPolicy.ANY

	@staticmethod
	def __check_returncode(run: Run, actual: Optional[int]) -> Verdict:
		policy = run.returncode_policy
		if policy is ReturnCodePolicy.ANY:
			return OK
		if policy is ReturnCodePolicy.SHOULD_BE_ZERO:
			if actual == 0:
				return OK
			return Verdict(VerdictCode.RETURN_CODE, f"expected 0, got {actual}")
		if policy is ReturnCodePolicy.SHOULD_NOT_BE_ZERO:
			if actual != 0:
				return OK
			return Verdict(VerdictCode.RETURN_CODE, "expected non-zero, got 0")
		if run.expected_returncode is None:
			return Verdict(
				VerdictCode.INTERNAL_ERROR,
				"MATCH_IF_PRESENTED requires expected_returncode",
			)
		if actual == run.expected_returncode:
			return OK
		return Verdict(
			VerdictCode.RETURN_CODE,
			f"expected {run.expected_returncode}, got {actual}",
		)

	@staticmethod
	def __check_dynamic_wrapper(runned: Runned) -> Verdict:
		log = runned.wrapper_log or ""
		if runned.dynamic_wrapper is DynamicWrapper.VALGRIND:
			summary = re.search(r"ERROR SUMMARY:\s*(\d+)", log)
			has_errors = summary is not None and int(summary.group(1)) != 0
			if runned.returncode == 125 or has_errors:
				return Verdict(
					VerdictCode.VALGRIND_ERROR,
					"memory errors were reported",
					tuple(log.splitlines()),
				)
		elif runned.dynamic_wrapper is DynamicWrapper.GDB:
			markers = (
				"Program received signal",
				"During startup program terminated with signal",
			)
			if any(marker in log for marker in markers):
				return Verdict(
					VerdictCode.GDB_ERROR,
					"the program terminated by a signal",
					tuple(log.splitlines()),
				)
		elif runned.dynamic_wrapper is DynamicWrapper.DR_MEMORY:
			summary = re.search(
				r"ERRORS FOUND:\s*(?:~~Dr\.M~~\s*)?([1-9]\d*)\s+unique",
				log,
				flags=re.IGNORECASE,
			)
			if runned.returncode == 125 or summary is not None:
				return Verdict(
					VerdictCode.DR_MEMORY_ERROR,
					"memory errors were reported",
					tuple(log.splitlines()),
				)
		elif runned.dynamic_wrapper is DynamicWrapper.WINDBG:
			if "TESTBUILD_FATAL_EXCEPTION" in log:
				return Verdict(
					VerdictCode.WINDBG_ERROR,
					"the program terminated with an unhandled exception",
					tuple(log.splitlines()),
				)
		return OK

	@staticmethod
	def __invoke_expected(expected: Expected, run: Run, runned: Runned) -> Verdict:
		try:
			result = expected.test(run, runned)
		except Exception as error:
			return Verdict(
				VerdictCode.EXPECTATION,
				f"{type(error).__name__}: {error}",
			)
		if result is None or result is True:
			return OK
		if isinstance(result, Verdict):
			return result
		if result is False:
			return Verdict(VerdictCode.EXPECTATION, "Expected.test returned False")
		return Verdict(VerdictCode.EXPECTATION, str(result))

	@staticmethod
	def __display(value: Union[bytes, str], encoding: str, limit: int = 240) -> str:
		rendered = repr(value if isinstance(value, str) else value.decode(encoding, errors="backslashreplace"))
		return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


__all__ = [
	"ANY",
	"EMPTY",
	"NONEMPTY",
	"NONZERO",
	"OK",
	"ActionContext",
	"DynamicWrapper",
	"Expected",
	"FileSource",
	"GroupBuilder",
	"HookBuilder",
	"MultiRunTestBuilder",
	"PolicyValue",
	"Report",
	"ReturnCodePolicy",
	"Run",
	"RunBuilder",
	"Runned",
	"StderrPolicy",
	"TestResult",
	"TestBuilder",
	"Tester",
	"Verdict",
	"VerdictCode",
	"from_file",
	"skip",
]
