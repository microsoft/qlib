import sys, platform
import qlib

def linux_distribution():
    try:
        return platform.linux_distribution()
    except:
        return "N/A"

print('Qlib version: {} \n'.format(qlib.__version__))
print("""Python version: {} \n
linux_distribution: {}
system: {}
machine: {}
platform: {}
version: {}
""".format(
sys.version.split('\n'),
linux_distribution(),
platform.system(),
platform.machine(),
platform.platform(),
platform.version(),
))