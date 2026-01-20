using OpenQA.Selenium;
using OpenQA.Selenium.Support.UI;
using SeleniumExtras.WaitHelpers;
namespace SauceTesting.Pages;

public class BasePage
{
    protected IWebDriver _driver;
    protected WebDriverWait wait;

    public BasePage(IWebDriver driver)
    {
        _driver = driver;
        wait = new WebDriverWait(driver, TimeSpan.FromSeconds(30));
    }

    protected void Click(By locator)
    {
        try
        {
            wait.Until(ExpectedConditions.ElementToBeClickable(locator)).Click();
        }
        catch (WebDriverTimeoutException)
        {
            try
            {
                var shortWait = new WebDriverWait(_driver, TimeSpan.FromSeconds(5));
                var element = shortWait.Until(ExpectedConditions.ElementExists(locator));

                ((IJavaScriptExecutor)_driver).ExecuteScript("arguments[0].click();", element);
            }
            catch (WebDriverTimeoutException)
            {
                throw new NoSuchElementException($"Element with locator '{locator}' was not found in DOM after retries.");
            }
        }
        catch (ElementClickInterceptedException)
        {
            var element = _driver.FindElement(locator);
            ((IJavaScriptExecutor)_driver).ExecuteScript("arguments[0].click();", element);
        }
    }

    protected void Type(By locator, string text)
    {
        var element = wait.Until(ExpectedConditions.ElementIsVisible(locator));
        element.Clear();
        element.SendKeys(text);
    }

    protected string GetText(By locator)
    {
        return wait.Until(ExpectedConditions.ElementIsVisible(locator)).Text;
    }

    protected bool IsVisible(By locator, int timeoutSeconds = 30)
    {
        var shortWait = new WebDriverWait(_driver, TimeSpan.FromSeconds(timeoutSeconds));
        try
        {
            shortWait.Until(ExpectedConditions.ElementIsVisible(locator));
            return true;
        }
        catch (WebDriverTimeoutException)
        {
            return false;
        }
    }
}