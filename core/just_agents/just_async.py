import asyncio
import threading
import concurrent.futures
from queue import Queue, Empty
from typing import Callable, AsyncGenerator, Any, Generator, Coroutine, TypeVar, Optional, Union

T = TypeVar('T')


def _run_coroutine_in_new_loop(coro: Coroutine[Any, Any, T], keep_loop_alive: bool = False) -> T:
    """
    Internal helper: Creates a new event loop, sets it for the current thread,
    runs the given coroutine in it, optionally keeps the loop alive, and returns the result.
    Exceptions from the coroutine are propagated.
    This function is designed to be the core logic within a function that is
    itself the target of a new thread.
    
    Args:
        coro: The coroutine to run
        keep_loop_alive: If True, don't close the loop (useful for daemon threads with reusable clients)
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        if not keep_loop_alive:
            loop.close()


def async_gen_to_sync(async_gen_factory: Callable[..., AsyncGenerator[Any, None]], *args: Any,
                      max_queue_size: int = 0,
                      queue_get_timeout: float | None = None,
                      **kwargs: Any) -> Generator[Any, None, None]:
    """
    Adapt an *async generator factory* (i.e. an async def that yields)
    into a regular, synchronous generator.

    ──────────────────────────────────────────────────────────────────────────────
    Parameters
    ----------
    async_gen_factory : Callable[…] -> AsyncGenerator
        A callable that returns the async generator when invoked.
    *args / **kwargs
        Forwarded to `async_gen_factory(*args, **kwargs)`.
    max_queue_size : int, default 0
        Size of the hand-off Queue (0 ⇒ unbounded).
    queue_get_timeout : float | None
        Seconds to wait for the next item before raising `StopIteration`
        (None ⇒ block indefinitely).
    Returns
    -------
    generator
        Iterate over it like a normal generator: `for item in ...: …`
    ──────────────────────────────────────────────────────────────────────────────

    The async generator is fully consumed in a separate daemon thread with its
    own event loop, guaranteeing:
        • No interference with a running loop in the calling thread.
        • No deadlocks caused by calling sync code from async context.
    """
    SENTINEL = object()           # marks normal completion
    EXC_FLAG = object()           # prefixes exceptions coming from the async side
    q: Queue = Queue(max_queue_size)

    # -------- background thread target ----------------------------------------
    async def _drain() -> None:
        """Helper coroutine to drain the async generator and put items/exceptions on the queue."""
        try:
            async for item in async_gen_factory(*args, **kwargs):
                q.put(item)
        except BaseException as exc:
            # Propagate exceptions to the consumer thread via the queue
            q.put((EXC_FLAG, exc))
        finally:
            q.put(SENTINEL)

    def _thread_worker() -> None:
        """Thread target that runs the _drain coroutine in a new event loop."""
        # _drain() creates the coroutine. 
        # _run_coroutine_in_new_loop handles its execution and loop lifecycle.
        # Exceptions from _drain (if any escape its own try/except, which is unlikely)
        # would propagate from _run_coroutine_in_new_loop and terminate the thread.
        # This is acceptable as _drain is designed to communicate all outcomes via the queue.
        _run_coroutine_in_new_loop(_drain())

    thread = threading.Thread(target=_thread_worker, daemon=True, name="AsyncGenWorker")
    thread.start()

    # -------- synchronous generator façade ------------------------------------
    def _sync_gen() -> Generator[Any, None, None]:
        while True:
            try:
                obj = q.get(timeout=queue_get_timeout)
            except Empty:
                raise StopIteration                     # configurable timeout
            if obj is SENTINEL:
                break                                   # normal termination
            if isinstance(obj, tuple) and obj and obj[0] is EXC_FLAG:
                raise obj[1] from None                 # re-raise async error
            yield obj

    return _sync_gen()


def run_async_function_synchronously(
    async_func: Callable[..., Coroutine[Any, Any, T]], 
    *args: Any, 
    timeout: Optional[float] = None,
    target_loop: Optional[asyncio.AbstractEventLoop] = None,
    **kwargs: Any
) -> T:
    """
    Runs an async function synchronously by executing it in a new event loop
    managed within a separate thread. This ensures that it doesn't interfere
    with any existing event loop in the calling thread.

    Parameters
    ----------
    async_func : Callable[..., Coroutine[Any, Any, T]]
        The asynchronous function (async def) to be executed.
    *args : Any
        Positional arguments passed to the `async_func`.
    timeout : Optional[float]
        Maximum time in seconds to wait for the function to complete.
        None means wait indefinitely. A TimeoutError is raised if the timeout is exceeded.

    **kwargs : Any
        Keyword arguments passed to the `async_func`.

    Returns
    -------
    T
        The result returned by the `async_func`.

    Raises
    ------
    TimeoutError
        If the function execution exceeds the specified timeout.
    BaseException
        Any exception raised by the `async_func` during its execution, including
        system exit, keyboard interrupts, and other non-Exception types.
    """
    # If a target loop is provided, schedule directly onto it and wait synchronously here
    if target_loop is not None:
        fut = asyncio.run_coroutine_threadsafe(async_func(*args, **kwargs), target_loop)
        return fut.result(timeout=timeout)

    # First, check if we're already in an event loop
    try:
        # If there's a running loop in this thread, we can't use run_until_complete
        # so we fall through to the thread-based approach
        loop = asyncio.get_running_loop()
        # If we get here, there's a running loop, so use the thread approach below
    except RuntimeError:
        # No running event loop, we can create a new one
        pass
    
    # No event loop running in this thread, use the thread approach
    result_container = [None]
    exception_container = [None]
    done_event = threading.Event()

    def _thread_worker() -> None:
        try:
            # Prepare the coroutine that needs to be run
            coro_to_run = async_func(*args, **kwargs)
            result_container[0] = _run_coroutine_in_new_loop(coro_to_run)
        except BaseException as e:
            # Capture ALL exceptions, including KeyboardInterrupt, SystemExit, etc.
            exception_container[0] = e
        finally:
            done_event.set()  # Signal completion regardless of success/failure

    thread = threading.Thread(target=_thread_worker, daemon=True, name="AsyncFuncRunner")
    thread.start()
    
    # Wait with optional timeout
    if not done_event.wait(timeout):
        thread.daemon = True  # Ensure the thread doesn't keep the process alive
        raise TimeoutError(f"Async function execution timed out after {timeout} seconds")
    
    # Re-raise any exception caught in the worker thread
    if exception_container[0] is not None:
        raise exception_container[0]
    
    return result_container[0]


