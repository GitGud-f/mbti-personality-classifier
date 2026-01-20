using OpenQA.Selenium;

namespace SauceTesting.Pages;

public class ProductDetailsPage : BasePage
{
    private By ProductTitle = By.CssSelector(".inventory_details_name");
    private By ProductPrice = By.CssSelector(".inventory_details_price");
    private By ProductDescription = By.CssSelector(".inventory_details_desc");
    private By BackButton = By.Id("back-to-products");
    private By InventoryActionButton = By.CssSelector(".btn_inventory"); 

    public ProductDetailsPage(IWebDriver driver) : base(driver) { }

    public string GetProductTitle() => GetText(ProductTitle);
    
    public decimal GetProductPrice()
    {
        string priceText = GetText(ProductPrice).Replace("$", "");
        return decimal.Parse(priceText);
    }

    public string GetProductDescription() => GetText(ProductDescription);

    public void ClickAddToCart() => Click(InventoryActionButton);

    public void ClickRemove() => Click(InventoryActionButton); 

    public void BackToProducts() => Click(BackButton);
    
    public bool IsRemoveButtonVisible() 
    {
        return GetText(InventoryActionButton).Equals("Remove", StringComparison.OrdinalIgnoreCase);
    }
}