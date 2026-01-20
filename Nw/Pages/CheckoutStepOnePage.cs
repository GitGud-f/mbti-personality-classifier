using OpenQA.Selenium;

namespace SauceTesting.Pages;

public class CheckoutStepOnePage : BasePage
{

    private By FirstNameField = By.Id("first-name");
    private By LastNameField = By.Id("last-name");
    private By PostalCodeField = By.Id("postal-code");
    private By ContinueButton = By.Id("continue");
    private By CancelButton = By.Id("cancel");
    private By ErrorMessage = By.CssSelector("[data-test='error']");
    
    public CheckoutStepOnePage(IWebDriver driver) : base(driver) { }

    public void EnterDetails(string firstName, string lastName, string zip)
    {
        Type(FirstNameField, firstName);
        Type(LastNameField, lastName);
        Type(PostalCodeField, zip);
    }

    public void Continue() => Click(ContinueButton);

    public void Cancel() => Click(CancelButton);

    public string GetErrorMessage() => GetText(ErrorMessage);
}
