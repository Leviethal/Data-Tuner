# 📊 Comparison of Missing Data Imputation Techniques

This report summarizes the findings from the evaluation of **statistical (Mean, Median)** and **machine learning-based (kNN)** imputation techniques under different levels of missing data.  
The performance was analyzed using **MAE (Mean Absolute Error)**, **L10**, **L25**, and **NOx Regression MSE** metrics.

---

## 🔹 1. Results from MAE, L10, and L25 Plots

### **MAE Trends**
- **Mean** and **Median** imputations show **almost constant MAE** across all levels of missingness, indicating that they are not sensitive to the proportion of missing data.  
- **kNN imputations (k=3, 5, 10)** yield **lower MAE** at low missingness levels (<10%) but MAE gradually increases as more data becomes missing.  
- Among kNN variants, **k=10 performs slightly better** at higher missingness (>50%).

### **L10 Performance**
- **kNN methods** achieve **L10 > 90%** at low missingness (<10%), meaning most imputed values are highly accurate.  
- As missingness increases, **L10 decreases sharply**, reflecting reduced precision in imputation.  
- **Mean** and **Median** have consistently low and flat L10 scores (~40–50%), showing less sensitivity but lower accuracy overall.

### **L25 Performance**
- All **kNN variants** achieve **L25 ≈ 99%** at low missingness, indicating that nearly all imputations are within acceptable error bounds.  
- L25 gradually decreases as missingness increases, but **kNN(10)** performs best, followed by **kNN(5)** and **kNN(3)**.  
- **Mean** and **Median** remain constant (~75–78%), showing stability but relatively low performance.

### ✅ **Inference**
- **kNN imputation consistently outperforms mean and median imputation**, especially as missingness increases.  
- **Statistical imputations (Mean/Median)** are reliable only for **low missingness (<10%)** scenarios.

---

## 🔹 2. Results from NOx Regression MSE Plot

- At **low missingness (<10%)**, **Mean** and **Median** imputations perform competitively, sometimes achieving the lowest MSE.  
- As missingness increases (≥20%), **kNN methods start outperforming** statistical ones in reducing regression MSE.  
- **kNN(3)** shows the **lowest MSE overall** beyond 20–30% missingness, providing a good balance between bias and variance.  
- **Mean** and **Median** MSE values increase steadily beyond 30–40% missingness, showing poor adaptability to higher data loss.

### ✅ **Inference**
- For **<10% missingness**, simple **Mean/Median** imputation is sufficient.  
- For **≥20% missingness**, **kNN (especially k=3–5)** provides more accurate and reliable regression performance.  
- **kNN remains stable** even at **high missingness (>50%)**, while statistical methods degrade noticeably.

---

## 🔹 3. Overall Summary

| Missingness Level | Recommended Method | Reason |
|--------------------|--------------------|--------|
| ≤ 10% | **Mean or Median Imputation** | Simple, efficient, and sufficient at low missingness. |
| ≥ 20% | **kNN (k=5 or k=10)** | Provides higher accuracy, lower MAE/MSE, and better stability. |

**Conclusion:**  
The **kNN imputation method** provides higher fidelity (high L10/L25) and lower prediction error (MSE), making it an ideal choice for handling **real-world datasets with moderate to high missingness**.
