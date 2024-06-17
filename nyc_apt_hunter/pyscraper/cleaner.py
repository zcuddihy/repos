from datetime import datetime
import re


def clean_unit_information(func):
    """
    A decorator to clean and format the scraped unit information.
    """

    def wrapper(self, *args, **kwargs):
        unit_information = func(self, *args, **kwargs)
        unit_information["rent"] = format_rent(unit_information["rent"])
        unit_information["beds"] = format_beds(unit_information["beds"])
        unit_information["baths"] = format_baths(unit_information["baths"])
        unit_information["sqft"] = format_sqft(unit_information["sqft"])
        unit_information["date_available"] = format_date_available(
            unit_information["date_available"]
        )
        return unit_information

    return wrapper


def format_rent(rent):
    try:
        rent = re.sub("[^0-9–]", "", rent)

        # Certain listing use a range for the posted rent
        if "–" in rent:
            low_value = int(rent.split("–")[0])
            high_value = int(rent.split("–")[-1])
            return (low_value + high_value) / 2
        else:
            return int(rent)
    except:
        return rent


def format_beds(beds):
    try:
        beds = beds.replace("Studio", "0")
        beds = re.sub("\D", "", beds)
        return beds
    except:
        return beds


def format_baths(baths):
    try:
        return re.sub("[^0-9.]", "", baths)
    except:
        return baths


def format_sqft(sqft):
    try:
        sqft = re.sub("[^0-9–]", "", sqft)

        # Certain listing use a range for the posted sqft
        if "–" in sqft:
            low_value = int(sqft.split("–")[0])
            high_value = int(sqft.split("–")[-1])
            return (low_value + high_value) / 2
        else:
            return int(sqft)
    except:
        return sqft


def format_date_available(date_available):
    date_available = (
        date_available.replace(r"Available Now", "Now")
        .replace(r"Available Soon", "Now")
        .replace("Soon", "Now")
    )
    try:
        if date_available == "Now":
            return datetime.now().strftime("%Y-%m-%d")
        else:
            current_year = datetime.now().year
            clean_date = (
                re.sub("[.]", "", date_available.split(",")[0]) + f", {current_year}"
            )
            date = datetime.strptime(clean_date, "%b %d, %Y")
            return date.strftime("%Y-%m-%d")
    except:
        return date_available
