from decimal import Decimal


def decimal_to_float(d: Decimal):
    """
    Correctly converts decimal to float without floating point errors.
    :param d: Decimal
    :return: Float
    """
    return round(float(d), 6)


def decimal_to_double(d: Decimal):
    """
    Correctly converts decimal to float without floating point errors.
    :param d: Decimal
    :return: Float with double precision
    """
    return round(float(d), 15)
