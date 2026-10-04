import os


def good_before():
    return 1


class Broken:
    def ok(self):
        return 1

    def bad(self:
        pass

    def also_ok(self):
        """Still here."""
        return 2


def good_after(x):
    """After the break."""
    return x


@decorator
async def good_async():
    pass
