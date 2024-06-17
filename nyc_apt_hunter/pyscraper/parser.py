from dataclasses import dataclass, field
import re
from cleaner import clean_unit_information
from datetime import date


@dataclass
class PropertyParser:
    property_name: str
    property_url: str
    description: str = field(init=False)
    amenities: dict = field(default_factory=dict)
    unique_features: list = field(default_factory=list)
    location: dict = field(default_factory=dict)
    year_built: str = field(init=False)

    def get_property_description(self, soup):
        raw_description = soup.find("section", {"id": "descriptionSection"}).find_all(
            "p"
        )
        # Extract all of the paragraphs
        paragraphs = []
        for paragraph in raw_description:
            paragraphs.append(paragraph.get_text(strip=True))

        raw_description = " ".join(paragraphs).replace("\n", "")
        raw_description = re.sub("[']", "", raw_description)
        raw_description = re.sub("[:]", "", raw_description)
        self.description = re.sub('"', "", raw_description)

    def extract_amenities(self, soup):
        """Parse the property HTML and search for various amenities"""

        # amenities to check for on the url
        # Key = the html element feature name
        # Value = the html element the value should be stored in
        html_amenities = {
            "Fitness Center": "span",
            "Business Center": "span",
            "Air Conditioning": "span",
            "In Unit Washer & Dryer": "span",
            "Dishwasher": "span",
            "Laundry Facilities": "span",
            "Car Charging": "span",
            "Roof": "span",
            "Concierge": "span",
            "Pool": "span",
            "Elevator": "span",
            "Garage": "div",
            "Dogs Allowed": "h4",
            "Cats Allowed": "h4",
            "Income Restrictions": "h2",
        }
        for amenity, html_element in html_amenities.items():
            amenity_check = (
                False
                if soup.find(html_element, text=re.compile(amenity)) == None
                else True
            )
            # The amenity name has to match the HTML lookup,
            # however, it needs to be converted to a proper format
            # for the database.
            cleaned_amenity_name = amenity.lower().replace(" & ", " ").replace(" ", "_")
            self.amenities[cleaned_amenity_name] = amenity_check

    def extract_unique_features(self, soup):
        """Parse the property HTML and collect a list of the unique features"""
        try:
            unique_features = soup.find("div", {"id": "uniqueFeatures"}).find_all(
                "li", {"class": "specInfo uniqueAmenity"}
            )
            # Extract the text of all the features
            for feature in unique_features:
                cleaned_feature = re.sub(
                    "[']", "", feature.get_text(strip=True))
                cleaned_feature = re.sub('"', "", cleaned_feature)
                self.unique_features.append(cleaned_feature)
        # Some listings don't have any unique features listed
        # find_all() will throw an error in this case
        except:
            self.unique_features = None

    def extract_location(self, soup):
        """Parse the property HTML and search for location amenities"""

        try:
            latitude = soup.find("meta", {"property": "place:location:latitude"})[
                "content"
            ]
            longitude = soup.find("meta", {"property": "place:location:longitude"})[
                "content"
            ]
        except:
            latitude = None
            longitude = None

        try:
            neighborhood = soup.find("a", {"class": "neighborhood"}).get_text(
                strip=True
            )
            zipcode = re.sub(
                "[\D]",
                "",
                soup.find("span", {"class": "stateZipContainer"}).get_text(
                    strip=True),
            )
        except:
            neighborhood = None
            zipcode = None

        self.location["latitude"] = latitude
        self.location["longitude"] = longitude
        self.location["neighborhood"] = neighborhood
        self.location["zipcode"] = zipcode

    def get_year_built(self, soup):
        try:
            year_built = soup.find("div", text=re.compile("Built in")).get_text(
                strip=True
            )
            self.year_built = re.sub("[\D]", "", year_built)
        except:
            self.year_built = None

    def parse_property_page(self, soup):
        self.get_property_description(soup)
        self.extract_amenities(soup)
        self.extract_unique_features(soup)
        self.extract_location(soup)
        self.get_year_built(soup)

        # Combined all of the dictionaries and other property values
        combined = (
            {"property_name": self.property_name} | self.amenities | self.location
        )
        combined["description"] = str(self.description)

        if self.unique_features != None:
            combined["unique_features"] = ", ".join(self.unique_features)
        else:
            combined["unique_features"] = None

        combined["year_built"] = self.year_built
        combined["property_url"] = str(self.property_url)
        return combined


@dataclass
class UnitParser:
    unit_label: str = field(init=False)
    rent: str = field(init=False)
    beds: str = field(init=False)
    baths: str = field(init=False)
    area: str = field(init=False)
    date_available: str = field(init=False)

    def get_unit_label(self, unit):
        try:
            self.unit_label = unit["data-unit"]
        except:
            self.unit_label = unit.find("span", {"class": "modelName"}).text

    def get_rent(self, unit):
        try:
            self.rent = (
                unit.find("div", {"class": "pricingColumn column"})
                .find_next()
                .find_next()
                .get_text(strip=True)
            )
        except:
            self.rent = unit.find("span", {"class": "rentLabel"}).text

        if self.rent == "":
            self.rent = None

    def get_bedrooms(self, unit):
        try:
            self.beds = unit["data-beds"]
        except:
            self.beds = (
                unit.find("span", {"class": "detailsTextWrapper"})
                .find_all("span")[0]
                .get_text()
            )

    def get_bathrooms(self, unit):
        try:
            self.baths = unit["data-beds"]
        except:
            self.baths = (
                unit.find("span", {"class": "detailsTextWrapper"})
                .find_all("span")[1]
                .get_text()
            )

    def get_sqft(self, unit):
        try:
            self.area = (
                unit.find("div", {"class": "sqftColumn column"})
                .get_text(strip=True)
                .strip("square feet")
            )
        except:
            self.area = (
                unit.find("span", {"class": "detailsTextWrapper"})
                .find_all("span")[2]
                .get_text()
            )
        if self.area == "":
            self.area = None

    def get_date_available(self, unit):
        try:
            self.date_available = (
                unit.find("span", {"class": "dateAvailable"})
                .get_text(strip=True)
                .strip("availability")
            )
        except AttributeError:
            self.date_available = unit.find(
                "span", {"class": "availabilityInfo"}
            ).get_text()
        except:
            self.date_available = None

    @clean_unit_information
    def parse_unit(self, unit):
        self.get_unit_label(unit)
        self.get_rent(unit)
        self.get_bedrooms(unit)
        self.get_bathrooms(unit)
        self.get_sqft(unit)
        self.get_date_available(unit)

        return {
            "unit_label": self.unit_label,
            "rent": self.rent,
            "beds": self.beds,
            "baths": self.baths,
            "sqft": self.area,
            "date_available": self.date_available,
            "date_scraped": date.today().strftime("%Y-%m-%d"),
        }
