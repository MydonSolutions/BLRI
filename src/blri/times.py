def julian_date_from_unix(unix):
    return unix/86400 + 2440587.5

def unix_from_julian_date(jd):
    return (jd - 2440587.5)*86400