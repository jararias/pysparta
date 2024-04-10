
def assert_that(cond, msg='', exc=None):
    if not cond:
        raise (exc or AttributeError)(msg)
