import asyncio
from playwright.async_api import async_playwright
import sys
import argparse
import random


async def control_browser(context_number):
  """Controls a single browser instance"""
  async with async_playwright() as p:
    # Launch browser in non-headless mode so we can see it
    browser = await p.chromium.launch(headless=False)

    # Calculate window position (5 rows x 5 columns grid)
    row = context_number // 5
    col = context_number % 5

    # Position window in a grid layout
    window_width = 400
    window_height = 600
    x_position = col * (window_width + 20)
    y_position = row * (window_height + 20)

    # Create context with specific window size
    context = await browser.new_context(
      viewport={"width": window_width, "height": window_height}
    )

    page = await context.new_page()

    # Position the window
    await page.evaluate(f"""
            window.moveTo({x_position}, {y_position});
            window.resizeTo({window_width}, {window_height});
        """)

    try:
      # Navigate to the app and wait for page load
      await page.goto("http://127.0.0.1:8080")
      print(f"Browser {context_number}: Connected")

      # Add initial random delay to stagger user starts
      await asyncio.sleep(random.uniform(0.1, 1.0))

      # Wait a second for any initial page setup
      await asyncio.sleep(1)

      # Keep pressing right arrow key
      while True:
        try:
          # Add random delay between actions (0.8 to 1.2 seconds)
          await asyncio.sleep(random.uniform(0.8, 1.2))

          # Check for "Experiment over" text
          if await page.locator("text='Experiment over'").is_visible():
            print(
              f"Browser {context_number}: Experiment over detected. Closing browser."
            )
            break  # Exit the loop to close the browser

          try:
            button = page.locator('button:has-text("START")')
            if await button.is_visible():
              await button.click()
              print(f"Browser {context_number}: Clicked start button")
          except:
            pass

          try:
            button = page.locator('button:has-text("NEXT")')
            if await button.is_visible():
              await button.click()
              print(f"Browser {context_number}: Clicked next button")

          except:
            pass

          try:
            button = page.locator('button:has-text("SUBMIT")')
            if await button.is_visible():
              await button.click()
              print(f"Browser {context_number}: Clicked submit button")

          except:
            pass

          await page.keyboard.press("ArrowRight")
          print(f"Browser {context_number}: Pressed right arrow")
        except Exception as e:
          print(f"Browser {context_number}: Error pressing key - {str(e)}")

    except Exception as e:
      print(f"Browser {context_number}: Error - {str(e)}")
    finally:
      await context.close()
      await browser.close()


async def main():
  # Set up command line argument parsing
  parser = argparse.ArgumentParser(description="Launch multiple browser instances")
  parser.add_argument(
    "-c",
    "--connections",
    type=int,
    default=20,
    help="Number of connections to launch (default: 20)",
  )
  args = parser.parse_args()

  print(f"Starting {args.connections} browser instances...")

  # Create tasks for all browsers
  tasks = []
  for i in range(args.connections):
    task = asyncio.create_task(control_browser(i))
    tasks.append(task)
    await asyncio.sleep(4)

  # Wait for all tasks to complete
  try:
    await asyncio.gather(*tasks)
  except KeyboardInterrupt:
    print("\nStopping all browsers...")
    sys.exit(0)


if __name__ == "__main__":
  asyncio.run(main())
