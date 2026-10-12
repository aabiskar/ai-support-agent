from strands_tools.browser import AgentCoreBrowser

browser = AgentCoreBrowser(region="us-east-1")

print(type(browser.browser))
print(browser.browser)
print("\nATTRIBUTES:")
print(dir(browser.browser))