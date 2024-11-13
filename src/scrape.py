# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

# def fetch_page_title(url):
#     # Set up the Selenium WebDriver to connect to the running selenium container
#     options = webdriver.ChromeOptions()
#     options.add_argument("--no-sandbox")
#     options.add_argument("--headless")  # Run in headless mode since we don't need a UI
#     options.add_argument("--disable-dev-shm-usage")

#     # Connect to the Selenium container
#     driver = webdriver.Remote(
#         command_executor='http://selenium:4444/wd/hub',  # URL of Selenium in docker-compose
#         options=options
#     )

#     # Open the webpage
#     driver.get(url)

#     # Get and print the page title
#     page_title = driver.title
#     print("Page title is:", page_title)

#     # Close the browser session
#     driver.quit()
    
#     return page_title

# from selenium import webdriver
# from selenium.webdriver.common.by import By

# def fetch_page_title(url: str):
#     options = webdriver.ChromeOptions()
#     options.add_argument("--no-sandbox")
#     # options.add_argument("--headless")
#     options.add_argument("--disable-dev-shm-usage")

#     driver = webdriver.Remote(
#         command_executor='http://selenium:4444/wd/hub',
#         options=options
#     )

#     try:
#         driver.get(url)

#         cookies = [
#             {
#                 "domain": ".flowreporter.com",
#                 "expirationDate": 1757163381,
#                 "httpOnly": False,
#                 "name": "_clck",
#                 "path": "/",
#                 "secure": False,
#                 "value": "t5ftre%7C2%7Cfoy%7C0%7C1698"
#             },
#             {
#                 "domain": ".flowreporter.com",
#                 "expirationDate": 1725714409,
#                 "httpOnly": False,
#                 "name": "_clsk",
#                 "path": "/",
#                 "secure": False,
#                 "value": "11zmybq%7C1725628009985%7C2%7C1%7Cu.clarity.ms%2Fcollect"
#             },
#             {
#                 "domain": "flowreporter.com",
#                 "httpOnly": True,
#                 "name": "auth0_session_0",
#                 "path": "/",
#                 "sameSite": "lax",
#                 "secure": False,
#                 "value": "%7B%22tag%22%3A%220ozcG7jOJjzQUzSDT%2B4OOQ%3D%3D%22%7D"
#             },
#             {
#                 "domain": "flowreporter.com",
#                 "httpOnly": True,
#                 "name": "auth0_session_1",
#                 "path": "/",
#                 "sameSite": "lax",
#                 "secure": False,
#                 "value": "C%2FwD9lbzNJ2Nr4j7IgwgMjtoJBKdy43Kyeitvu3a9AMmhNYM7f8oVAVfebKUb6j"
#             },
#             {
#                 "domain": "flowreporter.com",
#                 "expirationDate": 1727210404.677734,
#                 "httpOnly": False,
#                 "name": "CSRFToken",
#                 "path": "/",
#                 "secure": False,
#                 "value": "d9a8d2dae9632c3db838a3de1503680f54918f471fb8c333a626d081be6ae0c2"
#             },
#             {
#                 "domain": "flowreporter.com",
#                 "httpOnly": False,
#                 "name": "PHPSESSID",
#                 "path": "/",
#                 "secure": False,
#                 "value": "et0m24n82veqn92eunds81vb3r"
#             }
#         ]

#         # Add each cookie to the browser session
#         for cookie in cookies:
#             # Remove "expirationDate" as it's not valid for `add_cookie()`
#             cookie.pop("expirationDate", None)
#             driver.add_cookie(cookie)

#         # Reload the page after adding cookies
#         driver.get(url)

#         page_title = driver.title
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         page_title = None
#     finally:
#         driver.quit()
    
#     return page_title


from selenium import webdriver
from selenium.webdriver.common.by import By
import time

def fetch_page_info(url: str):
    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    # Comment out this line to enable viewing
    # options.add_argument("--headless")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Remote(
        command_executor='http://selenium:4444/wd/hub',
        options=options
    )

    try:
        driver.set_window_size(1280, 1024)  # Set browser window size
        driver.get(url)

        print("Page loaded. Adding cookies...")

        cookies = [
            {
                "domain": ".flowreporter.com",
                "httpOnly": False,
                "name": "_clck",
                "path": "/",
                "secure": False,
                "value": "t5ftre%7C2%7Cfpt%7C0%7C1698"
            },
            {
                "domain": ".flowreporter.com",
                "httpOnly": False,
                "name": "_clsk",
                "path": "/",
                "secure": False,
                "value": "16utcyj%7C1728271050491%7C4%7C1%7Ct.clarity.ms%2Fcollect"
            },
            {
                "domain": "flowreporter.com",
                "httpOnly": True,
                "name": "auth0_session_0",
                "path": "/",
                "secure": False,
                "value": "%7B%22tag%22%3A%22vFft%5C%2FBBhFZ4bmJg%5C%2FBJAHSg%3D%3D%22%2C%22iv%22%3A%22Qs%2BJ%5C%2Fc0wpPY9m7wC%22%2C%22data%22%3A%22eZ0yt8M0Y4Q5f35MlSa8FC7EAe%5C%2Fo0J0QZb5sGkcVFEsKHDyDoecBbW3b%2B7hwzq5vy0gd7MCiXkfxfJliNTs3E2aZ5irhJRmHgjl4YwUmcAttrjNBOuuZssLo5sHWwp2Os5WZCbXO1Ny%2BP9yRiFPO91iVchc7Koa8faPCDzMZ2Z6ppdhgmkW7P%5C%2FmUOY0gizF6Lf9msAd7JlYvdTuA%2BWo6031q%5C%2Fxcu1sVBnyuSDeloQaXvWHJ4hYyk9dPX3WibqLE9yLIevwqEzlemfYDGNUldHQURUelVTtpXGosHS74J8sNrVsCLU%2BV03cCP%2BqpJCBWal24KtCTIjDSUkQPurMzXO0qagWb8c0oL4zv0xBjZfAIiiliBLKTblbqgk6WOLUmGSBa0Kg2y0B2ijYFaN3tNa%5C%2FUsahW8fI5IaUn2MNRxisFQpGdTd5lflgz2xPSfhB%2BLWGYdfKswIS2lTf8r19VdOWyqxQs6TUfyjeoES2uBZoZIyn8MmVU3JoX9TuEnxORhBsS4zV0IIxcJa2biQgi2GaNnvPcFoFVY0sk7KLB%5C%2FG6YMFZGZ4WvD%2BaEWlUTscGpcb4lfVrpWmEmjIggvdioeXk%5C%2F0d9uzC4qQTGlI7Cxf%2BEouGUd6slfoBiaBPAbDPCQm5z3ABcYTP67VWjwYtIL8y8RVhaMMs%5C%2FljWDdK8CaWn4dWQKzZ1whHG2xOXHL7i5on9%2Bhp2HkG01rUsfUd25bxkArFxHAH7xgKc68tRTFqu52tkIJwd2B2NiITBCvOaQRu4v4W7wOHNcw78Z2IziudKcHPfINd9nYWzvlM%2BgArefzmBwm1uzcVlrcrAjwB%2BBxsZbp9rlZYdjCmDhn8VwgwsfvMxVQbcPYB8k2yUnF5Vr082qweUECPd7M7gzxKnzy9FVHMW4mYHUN7oaDtpx15VMhMFZjxdVEM06PAw2n41WsXnPO2P1MeImGpEn4%2BBac92oHkBAPCZp8o%5C%2FCEw%2BUIFPCBfbHCTux3fGli7shptn45f0DiuwtorlxH6wDChE0DtaaEe%5C%2FvJVvqh1uJI8LetU2fSJOZiaHUX9kDGWs5RZOKhZvnpuJZwRBUhf7yWT7y8EIGp33dfaOVnW8Dp7iN2Hxva%5C%2FmtCVqxM7wLMwfFqOy5ErQGJiogL3%2BUXttvZC%2Bh3TUyhfoNFWzjNtbONyJ8naEqZBfo3R5TfORa3QIr9o4GBg%2BLwAPOiZohrJs4i2%2BVY%5C%2FU71u%5C%2FE5ei0jFL1fu80bR2gGO115qFAz8EfqueqN1osp4GqFJQyE776%2BTzWf4cwISkH%2BQYwnD5oaTXkS4Y9PjG2GmI5yyIwqbfEZxpNG7LCHbzRRSw9lJDlTtJMN%5C%2F4qHpzR0JHNAhazjrYSQazDCMCXhtDzJKZRDS%5C%2FN90D3Lb66JbRD22C7Bi2b%5C%2FNpwhpG1hU1BzsHYujTLdPPUaf%2B0pMvtlsi%2BoaQ0Ie2dQ8EDZHgxgEsEdqHanyhiVbIiuji2pI1h%2Ban10p9l6k4KAqnC8ALtDotFBfwqkKtepsdPZ6ZVltFqAIangu%5C%2FkY%2B4clO2zOYXpUn7PUpCJgcehJ3VbK5cZMy3JYbLE%5C%2FTwdxyfQF76MnCvNsNZnoApv8raj8TnvkNT3OSyhDYf3M%2B7grJtDKD6zqPGq7PxX6QggOLVtu%2B8otIIhTk4TTfBe3urmdCKZKddlRSH6be%2B%2Bs73OfaTe9roJB8EgSeDAdvOZZtrX71xguoDPngtgurNbKeyhtN%2BNk7UUSBvwtCOmTgvRPfqd5IwtzQFLQm3FzfFPteZgMNWRwz%",
                    "id": 3
                },
                {
                    "domain": "flowreporter.com",
                    "httpOnly": True,
                    "name": "auth0_session_1",
                    "path": "/",
                    "secure": False,
                    "value": "2BWUc5Mxh6%2BrTp1rt%2Bv%5C%2Fn0AjP%5C%2Fi40sRAVGTXZbQTyLre5aMu5dtO2kFj5CIXwkXjzNugLJO3J5VgPtdL7YA5Ypbt23Y14a9loeyLEr9z1%2BxbVQ4TbM554m%5C%2FroGE%2Bm0AEVV2bGzy1WZINRLI7Mo46dKsS0cYu4zUtT5H8HW%2BUJLE8MNUXq0WQNKFYwXzJX2BVm2VpB%5C%2F1Z3buVFO9SzdYHdHgNP%5C%2F1y%2BVSK3%2B4NBpz9R%5C%2FzX1EmEPc49NXImLxUV4unuDFSAdGJ0twDxiDA1aB0x9AQQtdEeEKYZsIy01%5C%2FFuVSyHLVeBVAVN0elLyW3dDuzEvEqmyqj2%5C%2Fy9c4ILoVo6Qe%2BbLZXtVzm2HjhmDIscEimxwS%2BdKXvqwTDNbgrEMVOqe241dmqRT1H%2BhlbeSba3fFz7sdGmN4uq%5C%2FznoAI2pL2N%5C%2FQwpNJFuXp2YOtHl8K3eZIkZz2BZo4866xclUP9X6HELpVbv6Bc%2BY29qoxFttxUOMHdJrj%2BSDVw%2Bxzn3hthrZ6%5C%2FC6JKpl63plX8i2poi8CGSJvLIlCYRLCOyZZApJDBlGh%2BeggSiQS6GCiByCfqpNUAg4We3xTmmslVucsn7Tjy5mO9fDe3fLpZAsx5y4CxbMq6f7ZgvWOqTwKQpWggJ%2BluxGyaFCfMuhUFUq0cebfPLLUikR1OuSLygEZcXXLcOe1zOYZsCc%2ByPa5aHMt7hJ8fSPoyWFmPQU45mMSNH0%5C%2FUTjYn0eecuWYMOwfbqpaEF1phS8zBIgUOy4z578wCFS2MyPny9i0bx2lSVCdYltqClA4AXPrmrkeNw%2Bq7DKfH2TwAzVLnQXMann15Tn2ZjugVJXm4If%5C%2FoLxybUYQB9H1dPYfGKOnWlblEVJ55rFqddG1TQTYGIgeJ9FoIZgPyJqhqSP9%5C%2FM4sTTL5Y9hsCtVnSLGy4l0cpFoGqywfNpK%2BP36FBKEMEVLmDg7ITerKewbOMYf%2BjXO%2BOHZTW4M64OnW6%2BT5UUiiikqZlj3pLZsz4AQU327wyizfpw3NQIwQM74IVTqT2GdtGhCKvKZoYTz8aivekzy0xNEzDIuO%2BL2ZlI%5C%2FrN%2BRu1GXRshhvUznspMXzKVcJs%2B1%5C%2FiYgXsVqoD9vGA5DHRRvdlym77NvCg%2BGrtxq1S8droJlv7wVIpa1EqVrluStKFttD6%2Bjvz4gABY9CPlADDWN8pD6pSN76zYt19QoWiFEA1pGlay1dRXBuciY05QKJSBr0qfxhLXXLR8Go4ZYX9AF829YF9U9CF6j4NopNn8gZa0sWA4r9fNfNOXQmGdYg42Le0X9tQXcFDlznSLfc7pfIoBRS7d%5C%2FzqeNBkN0%2BI%5C%2F%5C%2Ft3aNJXwLPo8aSF5zoS5T2PsadMneOsexNKe36kmuho0zjZdNLoKbt%2B6kaFIVWtvc80WHB9du2byE4ebiLnrZaE1E%5C%2FjhdTifvF3WLfzRJEWAa2FezAk0vsibOLYb2kyN6B4HjE41pEjATCsY8HYN%2BgNWhWMt%2Bf7CkrqEDm4xXGUlsIZAUIrFz0ydxU2%2B6S2RcmbmQkgMAF6o36B1secHeD40xl3NON4gYY%2BLcRZ0yEopC%2BEUoVhXin93THGJmMk0UIG3ECEDTAqr%2Bm1P1bjsFxbKChM2bgpGtqZ%5C%2FQjKOCzG9tvqt8n7sW3ni317GbhaiY0PIGszpvjSiYiCTbAiUZn03UsvCCRfnt9wlTkxZGLSZ%2BS1ZxpBS4ZNBE7WjoTufD3A4aQVAdvHc5L3HfRUY%5C%2FBSUl9C5akqXDRdxBp36MeUydDhoA9dMdTskbT7telUVTO%2B09A7OpM6GjtLLARwR5boTU%2BJoxNscdlx%2BviJ75f3%5C%2FJ4B4UhhDR7Yp4W%2B96dHOAof%"
                },
                {
                    "domain": "flowreporter.com",
                    "httpOnly": True,
                    "name": "auth0_session_2",
                    "path": "/",
                    "secure": False,
                    "value": "22%7D"
                },
                {
                    "domain": "flowreporter.com",
                    "httpOnly": False,
                    "name": "CSRFToken",
                    "path": "/",
                    "secure": False,
                    "value": "9a9c02355c641b9777df6c2119ef81d78e442d6ed0540638d26e07267418213e68b94fdc777415d810993fa0f4481a4e479519b0d9153945f2e277f98341e97d"
                },
                # {
                #     "domain": "flowreporter.com",
                #     "httpOnly": False,
                #     "name": "PHPSESSID",
                #     "path": "/",
                #     "secure": False,
                #     "value": "t39ludmauvukda51epuut7raj1"
                # }
            ]

        for cookie in cookies:
            cookie.pop("expirationDate", None)
            driver.add_cookie(cookie)

        print("Cookies added. Reloading page...")
        driver.get(url)

        # Adding a delay to observe the scraping process
        time.sleep(5)

        dropdown = driver.find_element(By.CSS_SELECTOR, "g.highcharts-button.highcharts-contextbutton")

        dropdown.click()
        print("Dropdown clicked")

        time.sleep(1)

        element = driver.find_element(By.XPATH, "//li[contains(text(), 'Download CSV')]")
        element.click()
        print("Download CSV clicked")




        page_title = driver.title
        print(f"Page title: {page_title}")

    except Exception as e:
        print(f"An error occurred: {e}")
        page_title = None
    finally:
        time.sleep(5)  # Another delay before quitting to observe results
        driver.quit()
    
    return page_title
