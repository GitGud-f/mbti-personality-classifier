using OpenQA.Selenium;


namespace SauceTesting.Pages;

public class CheckoutStepTwoPage : BasePage
{
    private By SubtotalLabel = By.ClassName("summary_subtotal_label");
    private By TaxLabel = By.ClassName("summary_tax_label");
    private By TotalLabel = By.ClassName("summary_total_label");
    private By FinishButton = By.Id("finish");
    private By ItemPrices = By.ClassName("inventory_item_price");
    private By CancelButton = By.Id("cancel");

    public CheckoutStepTwoPage(IWebDriver driver) : base(driver) { }

    public void Finish() => Click(FinishButton);
    public void Cancel() => Click(CancelButton);

    private decimal ParsePrice(string text)
    {
        var priceText = text.Split('$').Last(); 
        return decimal.Parse(priceText);
    }

    public decimal GetSubtotal() => ParsePrice(GetText(SubtotalLabel));
    public decimal GetTax() => ParsePrice(GetText(TaxLabel));
    public decimal GetTotal() => ParsePrice(GetText(TotalLabel));

    public decimal SumOfVisibleItemPrices()
    {
        var prices = _driver.FindElements(ItemPrices)
            .Select(e => decimal.Parse(e.Text.Replace("$", "")))
            .ToList();
        return prices.Sum();
    }
}
