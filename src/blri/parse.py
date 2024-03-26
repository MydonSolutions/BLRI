
def degrees_process(value):
    if isinstance(value, str):
        units_factor = 1
        value_f = 0
        for part in value.split(':'):
            value_f += float(part)/units_factor
            if units_factor == 1:
                units_factor *= -1 if value_f < 0 else 1
            units_factor *= 60

        return value_f
    return float(value)

def to_sexagesimal(value:float, part_count=3) -> str:
    parts = []
    for i in range(part_count-1):
        mod_value = int(value)
        rem_value = value - mod_value
        value = abs(rem_value * 60)
        parts.append(f"{mod_value:02d}")
    parts.append(f"{value:02.6f}")
    return ":".join(parts)

if __name__ == "__main__":
    print(to_sexagesimal(34.07881373419933))
    print(to_sexagesimal(-107.61833419041476))