using OpenQA.Selenium;
using OpenQA.Selenium.Support.UI;
using SeleniumExtras.WaitHelpers;

namespace SauceTesting.Pages;

public class InventoryPage : BasePage
{
    private By PageTitle = By.ClassName("title");
    private By SortDropdown = By.ClassName("product_sort_container");
    private By InventoryItemNames = By.ClassName("inventory_item_name");
    private By InventoryItemPrices = By.ClassName("inventory_item_price");
    private By InventoryImages = By.CssSelector(".inventory_item_img img");
    private By InventoryContainer = By.Id("inventory_container");

    private By CartBadge = By.ClassName("shopping_cart_badge");
    private By CartLink = By.ClassName("shopping_cart_link");
    private By MenuButton = By.Id("react-burger-menu-btn");
    private By ResetAppStateLink = By.Id("reset_sidebar_link");
    private By CloseMenuButton = By.Id("react-burger-cross-btn");
    private By LogoutLink = By.Id("logout_sidebar_link");

    private By TwitterLink = By.LinkText("Twitter");
    private By FacebookLink = By.LinkText("Facebook");
    private By LinkedInLink = By.LinkText("LinkedIn");

    public InventoryPage(IWebDriver driver) : base(driver) { }

    public string GetPageTitle() => GetText(PageTitle);

    public void GoToCart() => Click(CartLink);

    public void SortBy(string sortOption)
    {
        SelectElement select = new SelectElement(_driver.FindElement(SortDropdown));
        select.SelectByText(sortOption);
    }

    public string GetActiveSortOption()
    {
        SelectElement select = new SelectElement(_driver.FindElement(SortDropdown));
        return select.SelectedOption.Text;
    }

    private static By GetItemButton(string productName)
    {
        string idParams = productName.ToLower().Replace(" ", "-");
        string addToCartId = $"add-to-cart-{idParams}";
        string removeId = $"remove-{idParams}";

        return By.CssSelector($"#{addToCartId}, #{removeId}");
    }

    public void AddToCart(string productName)
    {
        wait.Until(ExpectedConditions.ElementIsVisible(InventoryContainer));
        By buttonLocator = GetItemButton(productName);
        Click(buttonLocator);
    }

    public void RemoveFromCart(string productName)
    {
        By buttonLocator = GetItemButton(productName);
        Click(buttonLocator);
    }

    public void ResetAppState()
    {
        wait.Until(ExpectedConditions.ElementExists(MenuButton));
        Click(MenuButton);
        wait.Until(ExpectedConditions.ElementIsVisible(By.ClassName("bm-menu-wrap")));
        Click(ResetAppStateLink);
        Click(CloseMenuButton);
    }

    public void Logout()
    {
        Click(MenuButton);
        Click(LogoutLink);
    }

    public List<string> GetItemNames()
    {
        wait.Until(ExpectedConditions.VisibilityOfAllElementsLocatedBy(InventoryItemNames));
        return _driver.FindElements(InventoryItemNames).Select(e => e.Text).ToList();
    }

    public List<decimal> GetItemPrices()
    {
        wait.Until(ExpectedConditions.VisibilityOfAllElementsLocatedBy(InventoryItemPrices));
        return _driver.FindElements(InventoryItemPrices)
            .Select(e => decimal.Parse(e.Text.Replace("$", "")))
            .ToList();
    }
    public int GetCartItemCount()
    {
        try
        {
            return int.Parse(GetText(CartBadge));
        }
        catch (WebDriverTimeoutException)
        {
            return 0;
        }
    }

    public string GetImageSource(int itemIndex)
    {
        wait.Until(ExpectedConditions.ElementIsVisible(InventoryContainer));

        var images = _driver.FindElements(InventoryImages);
        if (itemIndex < images.Count)
        {
            return images[itemIndex].GetAttribute("src")!;
        }
        throw new IndexOutOfRangeException("Product index out of bounds");
    }

    public void OpenProductDetails(string productName)
    {
        var items = _driver.FindElements(InventoryItemNames);
        foreach (var item in items)
        {
            if (item.Text == productName)
            {
                item.Click();
                return;
            }
        }
        throw new NoSuchElementException($"Product with name '{productName}' not found.");
    }

    public string GetProductButtonText(string productName)
    {
        By buttonLocator = GetItemButton(productName);
        return GetText(buttonLocator);
    }

    public void ClickTwitter() => Click(TwitterLink);
    public void ClickFacebook() => Click(FacebookLink);
    public void ClickLinkedIn() => Click(LinkedInLink);
}