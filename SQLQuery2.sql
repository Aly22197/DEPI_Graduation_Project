--Top 5 Customers with the Most Positive Feedback in the Last Year
SELECT TOP 5 
    c.FirstName + ' ' + c.LastName AS CustomerName,
    COUNT(r.ReviewID) AS PositiveFeedbackCount,
    AVG(f.Rating) AS AverageRating
FROM FeedbackFact f
INNER JOIN CustomerDim c ON f.CustomerID = c.CustomerID
INNER JOIN ReviewDim r ON f.ReviewID = r.ReviewID
INNER JOIN DateDim d ON f.DateID = d.DateID
WHERE r.Sentiment = 'positive'
  AND d.FeedbackDate >= DATEADD(YEAR, -1, GETDATE())
GROUP BY c.FirstName, c.LastName
ORDER BY PositiveFeedbackCount DESC;

--Monthly Trends of Feedback Sentiments by Product Category
SELECT 
    d.Year,
    d.Month,
    p.ProductCategory,
    SUM(CASE WHEN r.Sentiment = 'positive' THEN 1 ELSE 0 END) AS PositiveCount,
    SUM(CASE WHEN r.Sentiment = 'neutral' THEN 1 ELSE 0 END) AS NeutralCount,
    SUM(CASE WHEN r.Sentiment = 'negative' THEN 1 ELSE 0 END) AS NegativeCount
FROM FeedbackFact f
INNER JOIN ProductDim p ON f.ProductID = p.ProductID
INNER JOIN ReviewDim r ON f.ReviewID = r.ReviewID
INNER JOIN DateDim d ON f.DateID = d.DateID
WHERE d.FeedbackDate >= DATEADD(MONTH, -12, GETDATE())
GROUP BY d.Year, d.Month, p.ProductCategory
ORDER BY d.Year, d.Month, p.ProductCategory;

--Correlation Between Total Amount Spent and Feedback Rating
SELECT 
    o.TotalAmount,
    AVG(f.Rating) AS AverageRating,
    COUNT(f.FeedbackID) AS FeedbackCount
FROM FeedbackFact f
INNER JOIN OrderDim o ON f.OrderID = o.OrderID
GROUP BY o.TotalAmount
HAVING COUNT(f.FeedbackID) > 10  -- Only consider cases with more than 10 feedback entries
ORDER BY o.TotalAmount;

--Feedback Rating Distribution Across Different Age Groups
SELECT 
    CASE 
        WHEN c.Age BETWEEN 18 AND 25 THEN '18-25'
        WHEN c.Age BETWEEN 26 AND 35 THEN '26-35'
        WHEN c.Age BETWEEN 36 AND 45 THEN '36-45'
        WHEN c.Age BETWEEN 46 AND 60 THEN '46-60'
        ELSE '60+' 
    END AS AgeGroup,
    AVG(f.Rating) AS AverageRating,
    COUNT(f.FeedbackID) AS FeedbackCount
FROM FeedbackFact f
INNER JOIN CustomerDim c ON f.CustomerID = c.CustomerID
GROUP BY 
    CASE 
        WHEN c.Age BETWEEN 18 AND 25 THEN '18-25'
        WHEN c.Age BETWEEN 26 AND 35 THEN '26-35'
        WHEN c.Age BETWEEN 36 AND 45 THEN '36-45'
        WHEN c.Age BETWEEN 46 AND 60 THEN '46-60'
        ELSE '60+' 
    END
ORDER BY AverageRating DESC;

--Average Order Amount for Different Sentiment Levels
SELECT 
    r.Sentiment,
    AVG(o.TotalAmount) AS AverageOrderAmount,
    COUNT(f.FeedbackID) AS NumberOfOrders
FROM FeedbackFact f
INNER JOIN OrderDim o ON f.OrderID = o.OrderID
INNER JOIN ReviewDim r ON f.ReviewID = r.ReviewID
GROUP BY r.Sentiment
ORDER BY AverageOrderAmount DESC;

--Detecting Seasonal Trends in Feedback Ratings
SELECT 
    d.Quarter,
    AVG(f.Rating) AS AverageRating,
    COUNT(f.FeedbackID) AS FeedbackCount
FROM FeedbackFact f
INNER JOIN DateDim d ON f.DateID = d.DateID
GROUP BY d.Quarter
ORDER BY d.Quarter;

--Products with the Most Consistent High Ratings
SELECT 
    d.Quarter,
    AVG(f.Rating) AS AverageRating,
    COUNT(f.FeedbackID) AS FeedbackCount
FROM FeedbackFact f
INNER JOIN DateDim d ON f.DateID = d.DateID
GROUP BY d.Quarter
ORDER BY d.Quarter;
