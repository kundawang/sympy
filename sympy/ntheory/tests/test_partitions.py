from __future__ import annotations
from sympy.ntheory.partitions_ import npartitions, _partition_rec, _partition


def test__partition_rec():
    A000041 = [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77, 101, 135,
               176, 231, 297, 385, 490, 627, 792, 1002, 1255, 1575]
    for n, val in enumerate(A000041):
        assert _partition_rec(n) == val


def test__partition():
    assert [_partition(k) for k in range(13)] == \
        [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77]
    assert _partition(100) == 190569292
    assert _partition(200) == 3972999029388
    assert _partition(1000) == 24061467864032622473692149727991
    assert _partition(1001) == 25032297938763929621013218349796
    assert _partition(2000) == 4720819175619413888601432406799959512200344166
    assert _partition(10000) % 10**10 == 6916435144
    assert _partition(100000) % 10**10 == 9421098519
    assert _partition(10000000) % 10**10 == 7677288980


def test_deprecated_ntheory_symbolic_functions():
    from sympy.testing.pytest import warns_deprecated_sympy

    with warns_deprecated_sympy():
        assert npartitions(0) == 1




def test__partition_thread_safety():
    # _partition uses the module-level _factor/_totient globals that are
    # assigned atomically by _pre so that concurrent calls always see
    # fully initialised values. Previously _pre assigned the globals
    # first and then filled them in place so another thread could read
    # partially initialised values.
    import sys
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import sympy.ntheory.partitions_ as partitions_mod

    # n must be large enough that _partition uses the HRR formula (with
    # _factor/_totient) rather than the _partition_rec recurrence.
    n = 15000
    expected = _partition(n)

    # Reset the globals so that the threads race to initialise them.
    for name in ('_factor', '_totient'):
        partitions_mod.__dict__.pop(name, None)

    nthreads = 4
    barrier = threading.Barrier(nthreads)

    def worker():
        barrier.wait(timeout=10)
        return _partition(n)

    # A small switch interval widens the window for races between threads.
    old_switchinterval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        with ThreadPoolExecutor(max_workers=nthreads) as executor:
            futures = [executor.submit(worker) for _ in range(nthreads)]
            results = [future.result() for future in futures]
    finally:
        sys.setswitchinterval(old_switchinterval)
    assert results == [expected]*nthreads
