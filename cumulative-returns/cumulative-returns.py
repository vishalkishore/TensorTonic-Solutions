def cumulative_returns(returns: list) -> list:
    """
    Returns the compounded cumulative return after every period.
    """
    wealth = 1.0
    ans = []

    for daily_return in returns:
        wealth = wealth * (1 + daily_return)
        ans.append(wealth - 1)

    return ans