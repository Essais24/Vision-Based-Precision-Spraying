import os
import time
import sys

# CORRECTED: pwmchip0 = platform/3280000.pwm = Pin 15's controller (confirmed
# via `sudo cat /sys/kernel/debug/pwm`), not Pin 32. Pin 32's actual
# controller is platform/32e0000.pwm = pwmchip3 on this board. Also:
# pwmchip0's npwm=1, meaning channel 7 was never valid there in the first
# place -- only channel 0 exists per chip on this SoC.
PWM_CHIP = "/sys/class/pwm/pwmchip3"
PWM_CHANNEL = "pwm0"
PATH = f"{PWM_CHIP}/{PWM_CHANNEL}"

# Target Timings in Nanoseconds (200Hz, 500us pulse)
PERIOD = 5000000
MIN_PULSE = 500000


def _try_write(path, node, value, label):
    """Best-effort write -- prints a warning and continues instead of
    raising. Needed because 'enable'/'duty_cycle' writes on a freshly
    exported channel (period still 0) legitimately fail with EINVAL until
    period has been set at least once; that's expected, not fatal."""
    try:
        with open(f"{path}/{node}", "w") as f:
            f.write(str(value))
        return True
    except OSError as e:
        print(f"{label}: {e}  (expected on a freshly-exported channel -- continuing)")
        return False


def run_hardware_pwm():
    # 1. Export channel 0 (not 7 -- see note above)
    if not os.path.exists(PATH):
        try:
            with open(f"{PWM_CHIP}/export", "w") as f:
                f.write("0")
            time.sleep(0.3)  # Wait for kernel folder population
        except IOError as e:
            print(f"Hardware Export Error: {e}")
            print("Tip: run `cat /sys/class/pwm/pwmchip3/npwm` first -- if "
                  "this errors or the path doesn't exist, pwmchip3 isn't "
                  "registered, which means Pin 32 isn't pinmux'd to PWM yet. "
                  "Re-run `sudo /opt/nvidia/jetson-io/jetson-io.py`, confirm "
                  "Pin 32 is set to PWM, save, and REBOOT (config alone "
                  "doesn't take effect without a reboot).")
            sys.exit(1)

    # 2-3. Best-effort clear (may legitimately EINVAL on a fresh channel --
    # that's fine, period below is what actually matters).
    _try_write(PATH, "enable", 0, "Pre-clear enable")
    _try_write(PATH, "duty_cycle", 0, "Pre-clear duty_cycle")

    try:
        # 4. Period MUST be set before duty_cycle/enable can succeed on a
        # freshly-exported channel -- this is the one write that has to work.
        with open(f"{PATH}/period", "w") as f: f.write(str(PERIOD))
        time.sleep(0.05)

        # 5. Now duty_cycle and enable will actually take.
        with open(f"{PATH}/duty_cycle", "w") as f: f.write(str(MIN_PULSE))
        time.sleep(0.05)

        with open(f"{PATH}/enable", "w") as f: f.write("1")
        print("SUCCESS: pwmchip3/pwm0 (Pin 32) is initialized.")
        print("Keep script running and measure physical Pin 32 now.")

    except OSError as e:
        print(f"Kernel Write Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_hardware_pwm()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nScript stopped.")
