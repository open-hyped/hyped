import asyncio
import multiprocessing as mp
import os
import pickle
from contextlib import nullcontext

import pytest

from hyped.common.lazy_instance import LazyInstance, LazyStaticInstance


def factory(pid=None) -> object:
    return "INSTANCE(%s)" % (pid or os.getpid())


def test_getitem() -> None:
    assert LazyStaticInstance(factory)[:4] == "INST"


@pytest.mark.asyncio
async def test_context_manager() -> None:
    with LazyStaticInstance(nullcontext):
        pass
    async with LazyStaticInstance(nullcontext):
        pass


class TestStaticLazyInstance(object):
    @pytest.fixture
    def obj(self) -> LazyStaticInstance:
        return LazyStaticInstance[str](factory)

    def test_pickel(self, obj) -> None:
        val = obj.lower()
        reconstructed = pickle.loads(pickle.dumps(obj))
        assert val == reconstructed.lower()

    def test_case(self, obj) -> None:
        assert not obj._is_instantiated()
        # interact with the object and check the instance
        assert obj.lower() == factory().lower()
        assert obj._is_instantiated()


class TestLazyInstance(TestStaticLazyInstance):
    @pytest.fixture
    def obj(self) -> LazyInstance:
        return LazyInstance[str](factory)

    def _mp_worker(self, obj: LazyInstance, value: object) -> None:
        # should have a different pid since the obj instance
        # is created within the worker process and the value
        # is coming from outside
        assert obj.lower() != value.lower()

    def test_same_instance(self) -> None:
        # check instance is not recreated when not needed to
        obj = LazyInstance(object)
        assert obj._get_instance() == obj._get_instance()

    def test_case_mp(self, obj: LazyInstance) -> None:
        p = mp.Process(
            target=self._mp_worker,
            args=(
                obj,
                factory(),
            ),
        )
        p.start()
        p.join()
        # check error in process
        assert p.exitcode == 0

    def test_case_new_loop(self, obj: LazyInstance) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # create instance
        obj.lower()
        loop_hash_A = object.__getattribute__(obj, "_loop_hash")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # create instance
        obj.lower()
        loop_hash_B = object.__getattribute__(obj, "_loop_hash")

        assert loop_hash_A != loop_hash_B
