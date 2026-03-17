"""Tests for core clock implementations (Step 02)."""
import pytest
from worldforge.core.clock import DiscreteClock
from worldforge.core.exceptions import WorldForgeError


class TestDiscreteClock:
    def test_initial_state(self):
        clock = DiscreteClock(steps=10)
        assert clock.now == 0
        assert not clock.is_done

    def test_ticks_correctly(self):
        clock = DiscreteClock(steps=5)
        for expected in range(1, 6):
            clock.tick()
            assert clock.now == expected

    def test_done_after_all_steps(self):
        clock = DiscreteClock(steps=3)
        assert not clock.is_done
        clock.tick()
        clock.tick()
        assert not clock.is_done
        clock.tick()
        assert clock.is_done

    def test_reset(self):
        clock = DiscreteClock(steps=5)
        clock.tick()
        clock.tick()
        clock.reset()
        assert clock.now == 0
        assert not clock.is_done

    def test_invalid_steps(self):
        with pytest.raises(ValueError):
            DiscreteClock(steps=0)
        with pytest.raises(ValueError):
            DiscreteClock(steps=-1)

    def test_single_step_clock(self):
        clock = DiscreteClock(steps=1)
        assert not clock.is_done
        clock.tick()
        assert clock.is_done
