using OpenQA.Selenium;

namespace SauceTesting.Pages;

public class CartPage : BasePage
{
    private By CartItemNames = By.ClassName("inventory_item_name");
    private By CheckoutButton = By.Id("checkout");
    private By ContinueShoppingButton = By.Id("continue-shopping");
    private By CartItemRemoveButtons = By.XPath("//button[text()='Remove']");

    public CartPage(IWebDriver driver) : base(driver) { }

    public List<string> GetCartItemNames()
    {
        if (!IsVisible(CartItemNames, 1)) return new List<string>();
        return _driver.FindElements(CartItemNames).Select(e => e.Text).ToList();
    }

    public void Checkout() => Click(CheckoutButton);
    
    public void ContinueShopping() => Click(ContinueShoppingButton);

    public void RemoveFirstItem()
    {
        var removeButtons = _driver.FindElements(CartItemRemoveButtons);
        if (removeButtons.Count > 0)
        {
            removeButtons[0].Click();
        }
    }
}
