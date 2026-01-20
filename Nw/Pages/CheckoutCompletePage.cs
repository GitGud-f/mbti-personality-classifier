using OpenQA.Selenium;


namespace SauceTesting.Pages;

public class CheckoutCompletePage : BasePage
{
    private By CompleteHeader = By.ClassName("complete-header");
    private By BackHomeButton = By.Id("back-to-products");

    public CheckoutCompletePage(IWebDriver driver) : base(driver) { }

    public string GetCompleteHeaderText() => GetText(CompleteHeader);
    
    public void BackHome() => Click(BackHomeButton);
}
