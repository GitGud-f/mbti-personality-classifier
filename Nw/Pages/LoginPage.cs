using OpenQA.Selenium;
using SeleniumExtras.WaitHelpers;
namespace SauceTesting.Pages;

public class LoginPage : BasePage
{
    private By UsernameField = By.Id("user-name");
    private By PasswordField = By.Id("password");
    private By LoginButton = By.Id("login-button");
    private By ErrorMessage = By.CssSelector("[data-test='error']");
    private By ErrorIcons = By.CssSelector(".error_icon");
    private By InputErrorState = By.CssSelector(".input_error");

    public LoginPage(IWebDriver driver) : base(driver) { }

    public void Login(string username, string password)
    {
        wait.Until(ExpectedConditions.ElementExists(LoginButton));
        Type(UsernameField, username);
        Type(PasswordField, password);
        Click(LoginButton);
    }

    public string GetErrorMessage() => GetText(ErrorMessage);

    public bool AreErrorIconsVisible() => IsVisible(ErrorIcons);

    public bool AreInputFieldsRed() => IsVisible(InputErrorState);

    public bool IsLoginButtonVisible() => IsVisible(LoginButton);

}