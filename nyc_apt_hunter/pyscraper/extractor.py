import numpy as np
from scraper import make_request, generate_page_URL
from parser import UnitParser, PropertyParser
import pickle as pkl
from datetime import date


class ApartmentsPipeline:
    """A class for extracting raw HTML from Apartments.com, parsing, and saving to a db

    Methods:
        get_property_urls: Extracts the general information for each listing
        get_property_urls_details: Extract the listing details for each listing
    """

    def __init__(
        self,
        city_name: str,
        state_abbv: str,
        start_price: int = 500,
        end_price: int = 15000,
        price_step: int = 250,
    ):
        """Constructs the attributes to use for web scraping apartments

        Args:
            city_name (str): City name to be scraped.
            state_abbv (str): Abbreviated state name to be scraped.
            db_file (str): Location of db for connection
            start_price (int, optional): Price to begin scraping. Defaults to 500.
            end_price (int, optional): Price to stop scraping. Defaults to 15000.
            price_step (int, optional): Sets the min and max price range. Defaults to 250.
        """

        self.start_price = int(start_price)
        self.end_price = int(end_price)
        self.price_step = int(price_step)
        self.city_name = city_name
        self.state_abbv = state_abbv
        # "/price range/page"
        self.BASE_URL = f"https://www.apartments.com/{city_name.lower().replace(' ', '-')}-{state_abbv.lower()}/"
        self.price_range = list(np.arange(start_price, end_price, price_step))
        self.page_range = list(np.arange(1, 29, 1))
        self.property_urls = []
        self.properties = []
        self.units = {}

    def get_property_urls(self):

        # Loop through all price ranges
        for min_price in self.price_range:

            for page_num in self.page_range:
                url = generate_page_URL(
                    self.BASE_URL,
                    min_price,
                    min_price + self.price_step,
                    page_num
                )
                soup, res_status = make_request(url)
                # If URL was redirected then end of page range was reached
                if res_status == 301:
                    break
                # If soup is None, skip iteration
                if soup is None:
                    continue

                listings = soup.find_all("li", {"class": "mortar-wrapper"})
                for listing in listings:
                    url = listing.find("a", {"class": "property-link"})["href"]
                    self.property_urls.append(url)

        # Remove any duplicate property URLs
        self.property_urls = list(set(self.property_urls))

    def scrape_property_urls(self):

        for url in self.property_urls:
            soup, res_status = make_request(url)
            # If soup is None, skip iteration
            if soup is None:
                continue
            try:
                property_name = (
                    soup.find("h1", {"class": "propertyName"})
                    .get_text(strip=True)
                    .replace("'", "")
                )

                # Extract all property details
                self.properties.append(
                    self.parse_property(soup,
                                        property_name,
                                        self.city_name,
                                        url
                                        )
                )
                # Find all of the current unit listings
                self.units[property_name] = {
                    "units": self.parse_units(soup),
                    "zipcode": self.properties[-1]["zipcode"],
                }

            except Exception as e:
                print(
                    f"The exception, {e}, occurred at the following URL: {url}")
                continue

    def parse_property(self, soup, property_name, city_name, url):
        property_data = PropertyParser(
            property_name, url).parse_property_page(soup)

        property_data["city_name"] = city_name

        return property_data

    def parse_units(self, soup):

        raw_units = self.get_all_units(soup)

        # Extract each unit from the listing
        units = []
        for unit in raw_units:
            current_unit = UnitParser().parse_unit(unit)
            if current_unit["date_available"] != "Not Available":
                units.append(current_unit)
            else:
                pass
        return units

    @staticmethod
    def get_all_units(soup):
        """Grab all units from a listing URl"""

        # Normal HTML layout for units
        units_html_1 = (soup
                        .find("div", {"data-tab-content-id": "all"})
                        .find_all("li", {"class": "unitContainer js-unitContainer"}
                                  ))

        # Alternate HTML layout for units
        units_html_2 = (soup
                        .find("div", {"data-tab-content-id": "all"})
                        .find_all("div", {"class": "pricingGridItem multiFamily"},
                                  ))

        units_html_3 = soup.find_all(
            "div", {"class": "priceGridModelWrapper js-unitContainer mortar-wrapper"},
        )
        if len(units_html_1) != 0:
            return units_html_1
        elif len(units_html_2) != 0:
            return units_html_2
        else:
            return units_html_3

    def run(self):
        print("Begin scraping...")
        self.get_property_urls()
        print("All property urls extracted")
        print(f"The total number of listings is {len(self.property_urls)}")
        self.scrape_property_urls()
        print("Done extracting properties and units")

        # Dump the saved data to have as a backup
        data_dump = [self.properties, self.units]
        scrape_date = date.today().strftime("%Y-%m-%d")
        filename = f"{self.city_name}_{scrape_date}.pkl"
        with open(f"./data/raw/{filename}", "wb") as file:
            pkl.dump(data_dump, file)


nyc_counties = [
    "Manhattan County",
    "Bronx County",
    "Brooklyn",
    "Queens County",
    "Staten Island",
]

if __name__ == "__main__":
    pipeline = ApartmentsPipeline('Manhattan County', 'NY')
    pipeline.run()
