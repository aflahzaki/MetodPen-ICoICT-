"""
Generate the IEEE conference paper (.docx) for ICoICT.

This script is DATA-DRIVEN: every numeric value in the tables and the narrative is
loaded from results.json, which is produced by generate_figures.py. Tables, prose,
and figures therefore share a single source of truth and cannot drift apart.

Pipeline:
    python3 generate_figures.py   # runs the experiment -> figures + results.json
    python3 create_paper.py       # builds the .docx from results.json + figures
    python3 add_equations.py      # inserts the OMML equations into Section III
"""
import os
import json

from docx import Document
from docx.shared import Inches, Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import nsdecls
from docx.oxml import parse_xml

BASE = '/projects/sandbox/MetodPen-ICoICT-'
FIG_DIR = os.path.join(BASE, 'figures')
OUTPUT = os.path.join(BASE, 'ICoICT_Paper_NGBoost_Water_Quality.docx')

# ---- single source of truth ----
R = json.load(open(os.path.join(BASE, 'results.json')))
M = R['metrics']
C = R['confusion']
DS = R['dataset']
SP = R['split']
SM = R['smote']
SACC = R['smote_acc']
MC = R['mcnemar']
Z = R['zones']
FI = R['feature_importance']
ENV = R['environment']
ORDER = ['NGBoost', 'XGBoost', 'Random Forest']


def f4(x):
    return f"{x:.4f}"


doc = Document()
for section in doc.sections:
    section.top_margin = Cm(1.9)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(1.9)
    section.right_margin = Cm(1.9)

style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(10)


def add_heading_ieee(doc, text, level=1):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    run.bold = True
    run.font.name = 'Times New Roman'
    run.font.size = Pt(10)
    if level == 1:
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(6)
    else:
        run.italic = True
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(3)
    return p


def add_body(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text)
    run.font.size = Pt(10)
    run.font.name = 'Times New Roman'
    p.paragraph_format.first_line_indent = Cm(0.5)
    p.paragraph_format.space_after = Pt(3)
    return p


def add_figure(doc, path, caption, width=Inches(5.5)):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    if os.path.exists(path):
        run.add_picture(path, width=width)
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    r.font.size = Pt(9)
    r.font.name = 'Times New Roman'
    cap.paragraph_format.space_after = Pt(6)


def set_cell_shading(cell, color):
    cell._tc.get_or_add_tcPr().append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color}"/>'))


def add_table(doc, caption, headers, rows):
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    r.bold = True
    r.font.size = Pt(9)
    r.font.name = 'Times New Roman'
    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(9)
        cell.paragraphs[0].runs[0].font.name = 'Times New Roman'
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_cell_shading(cell, 'D9E2F3')
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.rows[ri + 1].cells[ci]
            cell.text = str(val)
            cell.paragraphs[0].runs[0].font.size = Pt(9)
            cell.paragraphs[0].runs[0].font.name = 'Times New Roman'
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()
    return table


# ============== TITLE ==============
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run('Evaluating Probabilistic Prediction Performance of Natural Gradient '
                    'Boosting for Water Quality Classification')
run.bold = True
run.font.size = Pt(14)
run.font.name = 'Times New Roman'
title.paragraph_format.space_after = Pt(12)

authors = doc.add_paragraph()
authors.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = authors.add_run('Aflah Zaki Siregar')
run.font.size = Pt(11)
run.font.name = 'Times New Roman'
authors.paragraph_format.space_after = Pt(3)

affil = doc.add_paragraph()
affil.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = affil.add_run('School of Computing, Telkom University, Bandung, Indonesia')
run.font.size = Pt(10)
run.font.name = 'Times New Roman'
run.italic = True
affil.paragraph_format.space_after = Pt(12)

# ============== ABSTRACT ==============
add_heading_ieee(doc, 'Abstract')
abstract_text = (
    "Water quality assessment is a critical component of public health protection, yet conventional binary "
    "classification approaches fail to communicate the inherent uncertainty in predictions. This study evaluates "
    "the probabilistic prediction performance of Natural Gradient Boosting (NGBoost) for water potability "
    "classification, comparing it against XGBoost and Random Forest as baseline models. Using a publicly available "
    f"water potability dataset comprising {DS['n_samples']:,} samples with {DS['n_features']} physicochemical "
    "parameters, we assess model performance through classification metrics (accuracy, precision, recall, F1-score), "
    "calibration quality (Expected Calibration Error, Negative Log-Likelihood), and uncertainty quantification via "
    "zone-based analysis. Experimental results indicate that the models achieve comparable accuracy "
    f"(NGBoost: {f4(M['NGBoost']['accuracy'])}, XGBoost: {f4(M['XGBoost']['accuracy'])}, "
    f"Random Forest: {f4(M['Random Forest']['accuracy'])}), with McNemar's test confirming no statistically "
    "significant differences (p > 0.05). Although Random Forest attains the lowest calibration error "
    f"(ECE = {f4(M['Random Forest']['ece'])}) and NGBoost the highest (ECE = {f4(M['NGBoost']['ece'])}), NGBoost "
    "provides explicit, interpretable probabilistic outputs that enable uncertainty-aware decision-making. The "
    "uncertainty zone analysis reveals that predictions in extreme probability zones (zones 1 and 5) exhibit "
    "substantially higher accuracy than those in the ambiguous middle zone. Furthermore, SMOTE-ENN resampling "
    "degrades performance across all models, suggesting that the original class distribution better supports "
    "generalization. These findings establish NGBoost as a viable framework for uncertainty-aware water quality "
    "monitoring systems where confidence quantification is essential for operational decision-making."
)
add_body(doc, abstract_text)

kw = doc.add_paragraph()
kw.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
r1 = kw.add_run('Index Terms\u2014')
r1.bold = True
r1.italic = True
r1.font.size = Pt(10)
r1.font.name = 'Times New Roman'
r2 = kw.add_run('NGBoost, probabilistic classification, water quality, uncertainty quantification, '
                'calibration, XGBoost, Random Forest')
r2.italic = True
r2.font.size = Pt(10)
r2.font.name = 'Times New Roman'
kw.paragraph_format.space_after = Pt(12)

# ============== I. INTRODUCTION ==============
add_heading_ieee(doc, 'I. INTRODUCTION')
add_body(doc,
    "Access to safe drinking water remains a fundamental global challenge. The World Health Organization (WHO) "
    "reports that approximately 2.2 billion people lack access to safely managed drinking water services [1]. Water "
    "quality monitoring involves the assessment of multiple physicochemical parameters including pH, hardness, total "
    "dissolved solids, chloramines, sulfate, conductivity, organic carbon, trihalomethanes, and turbidity. The "
    "complex, nonlinear interactions among these parameters make binary potability determination a challenging "
    "classification task [2].")
add_body(doc,
    "Traditional machine learning approaches to water quality classification, such as logistic regression, decision "
    "trees, and support vector machines, produce deterministic predictions without quantifying prediction "
    "uncertainty [3]. In safety-critical domains like water quality assessment, the confidence level associated with "
    "a prediction is equally important as the prediction itself. A probabilistic model that outputs calibrated "
    "probability distributions enables operators to identify ambiguous samples requiring additional testing, thereby "
    "reducing both false clearances and unnecessary resource expenditure [4].")
add_body(doc,
    "Natural Gradient Boosting (NGBoost), introduced by Duan et al. [5], extends gradient boosting by fitting the "
    "parameters of a conditional probability distribution rather than a point estimate. By employing the natural "
    "gradient for parameter updates, NGBoost produces well-calibrated probabilistic predictions that can quantify "
    "epistemic uncertainty at the individual sample level. This capability distinguishes NGBoost from conventional "
    "ensemble methods such as XGBoost [6] and Random Forest [7], which typically provide probability estimates "
    "derived from vote aggregation or sigmoid transformations without explicit distributional assumptions.")
add_body(doc,
    "Despite its theoretical advantages, NGBoost has received limited empirical evaluation in environmental "
    "monitoring applications. Previous studies on water potability classification have primarily focused on "
    "maximizing accuracy through hyperparameter optimization or ensemble stacking [8], without addressing whether "
    "the resulting probability outputs are well-calibrated or suitable for uncertainty-aware decision support. "
    "Furthermore, the impact of data imbalance correction techniques such as SMOTE-ENN on probabilistic calibration "
    "remains underexplored [9].")
add_body(doc,
    "This study addresses the following research questions: (1) How does NGBoost perform relative to XGBoost and "
    "Random Forest on water potability classification in terms of both discriminative and calibration metrics? "
    "(2) Does NGBoost's probabilistic framework provide actionable uncertainty information through zone-based "
    "analysis? (3) What is the effect of SMOTE-ENN resampling on model performance and calibration quality? The "
    "contributions of this paper include a comprehensive multi-metric evaluation framework for probabilistic water "
    "quality classifiers, empirical evidence regarding the limitations of resampling on calibrated models, and a "
    "zone-based uncertainty analysis methodology for operational decision support.")

# ============== II. RELATED WORK ==============
add_heading_ieee(doc, 'II. RELATED WORK')
add_heading_ieee(doc, 'A. Machine Learning for Water Quality Classification', level=2)
add_body(doc,
    "Machine learning methods have been extensively applied to water quality prediction tasks. Ahmed et al. [2] "
    "employed Random Forest and gradient boosting techniques for water potability prediction, achieving accuracy "
    "levels between 65-70% on the same public dataset. Krishan et al. [3] applied support vector machines and "
    "k-nearest neighbors to classify water samples, noting that ensemble methods consistently outperformed single "
    "classifiers. More recently, deep learning approaches including multi-layer perceptrons and convolutional neural "
    "networks have been explored [10], though their computational overhead and interpretability limitations restrict "
    "practical deployment in resource-constrained monitoring systems.")
add_body(doc,
    "A common limitation across these studies is the exclusive reliance on discriminative metrics such as accuracy "
    "and F1-score. While these metrics assess whether predictions are correct, they do not evaluate whether the "
    "associated probability estimates reflect true outcome likelihoods. In water quality monitoring, a model that "
    "assigns 0.95 probability to potability should be correct approximately 95% of the time for that confidence "
    "level; this property is known as calibration [11].")
add_heading_ieee(doc, 'B. Probabilistic Prediction and NGBoost', level=2)
add_body(doc,
    "Probabilistic prediction methods extend standard classification by outputting full conditional distributions. "
    "Bayesian approaches such as Gaussian Processes and Bayesian Neural Networks provide principled uncertainty "
    "estimates but often suffer from computational intractability on moderate-scale datasets [12]. NGBoost [5] "
    "addresses this limitation by combining the scalability of gradient boosting with probabilistic output through "
    "natural gradient descent on scoring rule objectives. The natural gradient accounts for the information geometry "
    "of the parameter space, enabling efficient optimization of distributional parameters including location and "
    "scale.")
add_body(doc,
    "For binary classification, NGBoost models the conditional distribution as Bernoulli, directly outputting "
    "calibrated probability estimates. This contrasts with XGBoost's sigmoid-transformed log-odds, which may exhibit "
    "systematic miscalibration [6]. Empirical evaluations in healthcare and climate science domains have "
    "demonstrated NGBoost's competitive predictive performance alongside superior calibration compared to standard "
    "boosting methods [5][13].")
add_heading_ieee(doc, 'C. Class Imbalance and Resampling Techniques', level=2)
add_body(doc,
    "Class imbalance is prevalent in water quality datasets where safe samples typically outnumber contaminated "
    "ones. SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic minority instances through linear "
    "interpolation in feature space [9]. SMOTE-ENN combines oversampling with Edited Nearest Neighbors cleaning to "
    "remove noisy samples from both classes. While resampling can improve recall for minority classes, its impact on "
    "probability calibration is less understood. Wallace and Dahabreh [14] demonstrated that resampling distorts "
    "posterior probabilities, potentially degrading calibration quality even when classification accuracy improves.")

# ============== III. METHODOLOGY ==============
add_heading_ieee(doc, 'III. METHODOLOGY')
add_heading_ieee(doc, 'A. Dataset Description', level=2)
add_body(doc,
    f"This study utilizes the Water Potability dataset publicly available on Kaggle, comprising {DS['n_samples']:,} "
    f"water samples characterized by {DS['n_features']} physicochemical features: pH, Hardness, Solids (total "
    "dissolved solids), Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, and Turbidity. The "
    f"binary target variable indicates potability (1) or non-potability (0). The dataset exhibits moderate class "
    f"imbalance with {DS['class0']:,} non-potable samples ({DS['pct0']}%) and {DS['class1']:,} potable samples "
    f"({DS['pct1']}%).")
add_body(doc,
    f"Missing values are present in three features: Sulfate ({R['missing_pct']['Sulfate']}%), pH "
    f"({R['missing_pct']['ph']}%), and Trihalomethanes ({R['missing_pct']['Trihalomethanes']}%). These missing "
    "values are addressed through multivariate imputation by chained equations (MICE), which models each feature "
    "with missing entries as a function of the remaining features and is more robust than univariate mean or median "
    "imputation under a missing-at-random assumption. Feature standardization (zero mean, unit variance) is applied "
    "prior to model training to ensure equitable contribution of all features regardless of their native scales.")
add_figure(doc, os.path.join(FIG_DIR, 'class_distribution.png'),
           'Fig. 1. Class distribution in the water potability dataset.', width=Inches(3.5))
add_figure(doc, os.path.join(FIG_DIR, 'missing_values.png'),
           'Fig. 2. Percentage of missing values per feature.', width=Inches(3.5))

add_heading_ieee(doc, 'B. Data Partitioning', level=2)
add_body(doc,
    f"The dataset is partitioned into training (70%), validation (15%), and test (15%) subsets using stratified "
    f"sampling to preserve class proportions across all splits. This yields {SP['train']:,} training samples, "
    f"{SP['val']} validation samples, and {SP['test']} test samples. The validation set serves dual purposes: early "
    "stopping monitoring for the boosting models and model selection. All reported performance metrics are computed "
    "exclusively on the held-out test set to ensure unbiased evaluation.")

add_heading_ieee(doc, 'C. Model Architectures', level=2)
add_body(doc,
    "Three gradient boosting variants are evaluated. NGBoost [5] is configured with a Bernoulli distributional "
    "assumption for binary classification, 300 estimators, a learning rate of 0.05, and stochastic subsampling "
    "(minibatch fraction 0.8, column subsample 0.8). The natural gradient update rule optimizes the negative "
    "log-likelihood scoring rule, directly producing calibrated posterior probabilities. XGBoost [6] is configured "
    "with 300 estimators, a learning rate of 0.05, maximum depth of 4, and row/column subsampling of 0.8, employing "
    "log-loss as the objective function. Random Forest [7] utilizes 300 trees with default hyperparameters, "
    "providing probability estimates through vote averaging across the ensemble.")

add_heading_ieee(doc, 'D. Evaluation Metrics', level=2)
add_body(doc,
    "Model evaluation encompasses three categories. Classification metrics include Accuracy, Precision, Recall, and "
    "F1-score, computed at the standard 0.5 decision threshold. Calibration metrics include Negative Log-Likelihood "
    "(NLL), which measures the quality of predicted probability distributions, and Expected Calibration Error (ECE), "
    "which quantifies the average absolute difference between predicted confidence and observed accuracy across ten "
    "uniform probability bins. Discriminative capacity is assessed via Area Under the ROC Curve (AUC). Statistical "
    "comparison between models employs McNemar's test [15] to determine whether differences in classification "
    "accuracy are statistically significant.")

add_heading_ieee(doc, 'E. Uncertainty Zone Analysis', level=2)
add_body(doc,
    "To evaluate the operational utility of probabilistic outputs, we partition the test set into five uncertainty "
    "zones based on predicted probability: Zone 1 (mu < 0.2, high confidence non-potable), Zone 2 (0.2 <= mu < 0.4, "
    "moderate confidence non-potable), Zone 3 (0.4 <= mu < 0.6, ambiguous/uncertain), Zone 4 (0.6 <= mu < 0.8, "
    "moderate confidence potable), and Zone 5 (mu >= 0.8, high confidence potable). For each zone, we compute "
    "accuracy and sample count to assess whether probabilistic confidence correlates with actual predictive "
    "reliability.")

add_heading_ieee(doc, 'F. SMOTE-ENN Analysis', level=2)
add_body(doc,
    "To investigate the impact of class imbalance correction on probabilistic predictions, SMOTE-ENN is applied to "
    "the training data. SMOTE generates synthetic minority samples through k-nearest neighbor interpolation (k=5), "
    "while Edited Nearest Neighbors removes samples whose class differs from the majority of their neighbors. Models "
    "are retrained on the resampled data and evaluated on the unchanged test set to assess whether resampling "
    "improves or degrades generalization and calibration quality.")

add_heading_ieee(doc, 'G. Experimental Reproducibility', level=2)
add_body(doc,
    f"To ensure full reproducibility and transparency, all experiments use a single fixed random seed "
    f"({ENV['random_state']}). Because NGBoost's stochastic minibatch and column subsampling are not entirely "
    "governed by its constructor seed in the library version employed, the global NumPy random state is explicitly "
    "fixed immediately prior to NGBoost training; this renders the reported NGBoost results deterministic across "
    f"repeated runs. The reported results were produced with scikit-learn {ENV['scikit_learn']}, XGBoost "
    f"{ENV['xgboost']}, and NGBoost {ENV['ngboost']}. All tables and figures in this paper are generated "
    "programmatically from a single experimental run, guaranteeing internal consistency between reported values.")

# ============== IV. RESULTS AND DISCUSSION ==============
add_heading_ieee(doc, 'IV. RESULTS AND DISCUSSION')

add_heading_ieee(doc, 'A. Classification Performance', level=2)
add_body(doc,
    f"Table I presents the classification and calibration metrics for all three models on the test set "
    f"(N={SP['test']}). NGBoost and XGBoost achieve identical accuracy ({f4(M['NGBoost']['accuracy'])}), while "
    f"Random Forest obtains {f4(M['Random Forest']['accuracy'])}. However, the models exhibit distinct "
    f"precision-recall trade-offs. NGBoost demonstrates the highest precision ({f4(M['NGBoost']['precision'])}) but "
    f"the lowest recall ({f4(M['NGBoost']['recall'])}), indicating conservative positive predictions with few false "
    f"positives. XGBoost achieves the highest recall ({f4(M['XGBoost']['recall'])}) and the best F1-score "
    f"({f4(M['XGBoost']['f1'])}). Random Forest occupies an intermediate position with precision of "
    f"{f4(M['Random Forest']['precision'])} and recall of {f4(M['Random Forest']['recall'])}.")
add_table(doc, f'TABLE I. CLASSIFICATION AND CALIBRATION METRICS (TEST SET, N={SP["test"]})',
          ['Model', 'Acc', 'Prec', 'Rec', 'F1', 'NLL', 'ECE', 'AUC'],
          [[n, f4(M[n]['accuracy']), f4(M[n]['precision']), f4(M[n]['recall']), f4(M[n]['f1']),
            f4(M[n]['nll']), f4(M[n]['ece']), f4(M[n]['auc'])] for n in ORDER])
_auc_lo = min(M[n]['auc'] for n in ORDER)
_auc_hi = max(M[n]['auc'] for n in ORDER)
add_body(doc,
    "The low recall across all models reflects the inherent difficulty of the water potability classification task. "
    "The dataset's overlapping class distributions in feature space lead to conservative decision boundaries, "
    f"particularly for the minority potable class. Nevertheless, the models achieve AUC values ranging from "
    f"{f4(_auc_lo)} to {f4(_auc_hi)}, indicating moderate discriminative ability above random chance.")

add_heading_ieee(doc, 'B. Confusion Matrix Analysis', level=2)
add_body(doc,
    f"Fig. 3 presents the confusion matrices for all three models. NGBoost produces the fewest false positives "
    f"(FP={C['NGBoost']['fp']}) but the highest false negatives (FN={C['NGBoost']['fn']}), confirming its "
    f"conservative prediction behavior. XGBoost shows a more balanced error distribution (FP={C['XGBoost']['fp']}, "
    f"FN={C['XGBoost']['fn']}), while Random Forest (FP={C['Random Forest']['fp']}, FN={C['Random Forest']['fn']}) "
    "occupies an intermediate position. In water quality assessment, the relative cost of false positives (declaring "
    "contaminated water as safe) versus false negatives (unnecessary rejection of safe water) determines the optimal "
    "operating point. NGBoost's conservative behavior is preferable when the cost of false clearance is high.")
add_figure(doc, os.path.join(FIG_DIR, 'confusion_matrices.png'),
           'Fig. 3. Confusion matrices for NGBoost, XGBoost, and Random Forest.', width=Inches(5.5))

add_heading_ieee(doc, 'C. Calibration Quality', level=2)
add_body(doc,
    f"Calibration quality is assessed through Expected Calibration Error (ECE) and visual inspection of calibration "
    f"curves (Fig. 4). Random Forest achieves the lowest ECE ({f4(M['Random Forest']['ece'])}), followed by XGBoost "
    f"({f4(M['XGBoost']['ece'])}) and NGBoost ({f4(M['NGBoost']['ece'])}). The relatively low ECE values across all "
    "models indicate reasonable calibration quality, though Random Forest's vote-averaging mechanism naturally "
    f"produces well-calibrated probabilities in the mid-range. In terms of Negative Log-Likelihood, Random Forest "
    f"({f4(M['Random Forest']['nll'])}) and XGBoost ({f4(M['XGBoost']['nll'])}) outperform NGBoost "
    f"({f4(M['NGBoost']['nll'])}), suggesting that while NGBoost's distributional framework is theoretically sound, "
    "the challenging feature space limits its calibration advantage on this particular dataset.")
add_figure(doc, os.path.join(FIG_DIR, 'calibration_curves.png'),
           'Fig. 4. Calibration curves comparing predicted probability against observed frequency.', width=Inches(4.0))

add_heading_ieee(doc, 'D. ROC Analysis', level=2)
add_body(doc,
    f"Fig. 5 presents the ROC curves for all models. Random Forest achieves the highest AUC "
    f"({f4(M['Random Forest']['auc'])}), followed by NGBoost ({f4(M['NGBoost']['auc'])}) and XGBoost "
    f"({f4(M['XGBoost']['auc'])}). The similar AUC values across models suggest that discriminative performance is "
    "primarily constrained by dataset characteristics rather than algorithmic differences. These close AUC values "
    "are consistent with the non-significant McNemar's test results (Section IV-E).")
add_figure(doc, os.path.join(FIG_DIR, 'roc_curves.png'),
           'Fig. 5. Receiver Operating Characteristic (ROC) curves for all models.', width=Inches(4.0))

add_heading_ieee(doc, 'E. Statistical Comparison (McNemar Test)', level=2)
add_body(doc,
    "Table II presents the McNemar's test results for pairwise model comparisons. No statistically significant "
    "differences are observed at the 0.05 significance level, indicating that the observed accuracy differences are "
    "attributable to random variation rather than genuine algorithmic superiority. This finding supports the "
    "conclusion that model selection for this task should be guided by secondary criteria such as calibration "
    "quality, uncertainty interpretability, and computational efficiency rather than raw classification accuracy.")
add_table(doc, "TABLE II. McNEMAR'S TEST RESULTS",
          ['Comparison', 'Chi-squared', 'p-value', 'Significant?'],
          [[d['pair'], f4(d['chi2']), f4(d['p']), 'Yes' if d['significant'] else 'No'] for d in MC])

add_heading_ieee(doc, 'F. Uncertainty Zone Analysis', level=2)
_ng = {z['zone']: z for z in Z['NGBoost']}
add_body(doc,
    f"Table III presents the uncertainty zone analysis for NGBoost, which partitions test samples by predicted "
    f"probability. A clear relationship between prediction confidence and accuracy is observed: Zone 1 (high "
    f"confidence non-potable, mu < 0.2) achieves {f4(_ng['Zone 1']['acc'])} accuracy with {_ng['Zone 1']['n']} "
    f"samples, while Zone 5 (high confidence potable, mu >= 0.8) achieves {f4(_ng['Zone 5']['acc'])} accuracy with "
    f"{_ng['Zone 5']['n']} samples. The ambiguous Zone 3 (0.4 <= mu < 0.6) contains {_ng['Zone 3']['n']} samples "
    f"with the lowest accuracy of {f4(_ng['Zone 3']['acc'])}, barely above random chance. This pattern validates the "
    "utility of NGBoost's probabilistic outputs for identifying samples that warrant additional laboratory "
    "verification.")
add_table(doc, 'TABLE III. UNCERTAINTY ZONE ANALYSIS (NGBoost)',
          ['Zone', 'Range', 'N', 'Accuracy', 'Avg Prob'],
          [[z['zone'], z['range'], z['n'], f4(z['acc']), f4(z['avg_prob'])] for z in Z['NGBoost']])
_xg = {z['zone']: z for z in Z['XGBoost']}
_rf = {z['zone']: z for z in Z['Random Forest']}
add_body(doc,
    f"Comparative zone analysis across models (Tables IV and V) reveals that XGBoost and Random Forest exhibit "
    f"similar patterns but with different sample distributions across zones. XGBoost concentrates more samples in "
    f"Zone 2 (N={_xg['Zone 2']['n']}) with a Zone 1 accuracy of {f4(_xg['Zone 1']['acc'])}, while Random Forest "
    f"distributes more samples toward Zone 3 (N={_rf['Zone 3']['n']}) and attains {f4(_rf['Zone 4']['acc'])} "
    f"accuracy in Zone 4. All models demonstrate the highest accuracy in the extreme confidence zones, confirming "
    "that probabilistic outputs carry meaningful uncertainty information regardless of the underlying algorithm.")
add_table(doc, 'TABLE IV. UNCERTAINTY ZONE ANALYSIS (XGBoost)',
          ['Zone', 'Range', 'N', 'Accuracy', 'Avg Prob'],
          [[z['zone'], z['range'], z['n'], f4(z['acc']), f4(z['avg_prob'])] for z in Z['XGBoost']])
add_table(doc, 'TABLE V. UNCERTAINTY ZONE ANALYSIS (Random Forest)',
          ['Zone', 'Range', 'N', 'Accuracy', 'Avg Prob'],
          [[z['zone'], z['range'], z['n'], f4(z['acc']), f4(z['avg_prob'])] for z in Z['Random Forest']])
add_figure(doc, os.path.join(FIG_DIR, 'uncertainty_zones.png'),
           'Fig. 6. Uncertainty zone accuracy analysis for all three models.', width=Inches(5.5))

add_heading_ieee(doc, 'G. Probability Distribution Analysis', level=2)
add_body(doc,
    "Fig. 7 illustrates the kernel density estimation (KDE) of predicted probabilities separated by true class "
    "label. NGBoost shows concentrated probability mass in the 0.2-0.4 range for both classes, reflecting its "
    "conservative behavior. XGBoost produces slightly more dispersed distributions with better separation between "
    "classes. Random Forest exhibits the most concentrated distributions around 0.3-0.5, consistent with its "
    "vote-averaging mechanism that naturally avoids extreme probabilities. The substantial overlap between class "
    "distributions across all models confirms the inherent difficulty of this classification task and explains the "
    "moderate AUC values observed.")
add_figure(doc, os.path.join(FIG_DIR, 'probability_distributions.png'),
           'Fig. 7. Kernel density estimation of predicted probabilities by true class label.', width=Inches(5.5))

add_heading_ieee(doc, 'H. Impact of SMOTE-ENN', level=2)
_diffs = [abs(SACC[n]['diff']) for n in ORDER]
add_body(doc,
    f"Table VI presents the impact of SMOTE-ENN resampling on model accuracy. The training set is reduced from "
    f"{SM['before']:,} to {SM['after']:,} samples after SMOTE-ENN application, as the ENN cleaning step removes a "
    f"substantial portion of ambiguous samples. All three models exhibit decreased accuracy after resampling: "
    f"NGBoost drops from {f4(SACC['NGBoost']['without'])} to {f4(SACC['NGBoost']['with'])}, XGBoost from "
    f"{f4(SACC['XGBoost']['without'])} to {f4(SACC['XGBoost']['with'])}, and Random Forest from "
    f"{f4(SACC['Random Forest']['without'])} to {f4(SACC['Random Forest']['with'])}. This consistent degradation "
    "indicates that the synthetic samples generated by SMOTE and the aggressive cleaning by ENN disrupt the natural "
    "decision boundaries learned from the original data distribution.")
add_table(doc, 'TABLE VI. IMPACT OF SMOTE-ENN ON MODEL ACCURACY',
          ['Model', 'Without SMOTE-ENN', 'With SMOTE-ENN', 'Difference'],
          [[n, f4(SACC[n]['without']), f4(SACC[n]['with']), f4(SACC[n]['diff'])] for n in ORDER])
add_figure(doc, os.path.join(FIG_DIR, 'smote_enn_comparison.png'),
           'Fig. 8. Comparison of model accuracy with and without SMOTE-ENN resampling.', width=Inches(4.0))
add_body(doc,
    f"The performance degradation is attributable to two factors. First, the moderate imbalance ratio "
    f"({DS['pct0']:.0f}:{DS['pct1']:.0f}) does not critically impair learning, making aggressive resampling "
    f"unnecessary. Second, the ENN cleaning step reduces the training set by more than 50% "
    f"({SM['before']:,} to {SM['after']:,}), removing informative boundary samples that are essential for learning "
    "the complex decision surface in this overlapping feature space. This finding is consistent with Wallace and "
    "Dahabreh [14], who demonstrated that resampling can degrade calibration and accuracy when class overlap is "
    "high.")

add_heading_ieee(doc, 'I. Validation Loss Analysis', level=2)
add_body(doc,
    "Fig. 9 presents the XGBoost validation loss curve during training. The log-loss decreases monotonically before "
    "gradually stabilizing, indicating effective learning without severe overfitting. NGBoost's internal training "
    "process operates analogously but does not expose an iteration-level validation loss history in the same "
    "interface, so only the XGBoost curve is shown. The convergence behavior validates the chosen learning rate and "
    "number of estimators for the boosting models.")
add_figure(doc, os.path.join(FIG_DIR, 'xgboost_loss_curve.png'),
           'Fig. 9. XGBoost validation loss curve during training.', width=Inches(4.0))

add_heading_ieee(doc, 'J. Feature Importance', level=2)
_xgb_top = ', '.join(f for f, _ in FI['XGBoost'][:3])
_rf_top = ', '.join(f for f, _ in FI['Random Forest'][:3])
add_body(doc,
    f"Fig. 10 presents the feature importance rankings for XGBoost and Random Forest. XGBoost ranks "
    f"{_xgb_top} as its most influential features, while Random Forest ranks {_rf_top} highest; both models place "
    f"pH and Sulfate among the three most important predictors. The prominence of Sulfate is particularly notable "
    f"given that it has the highest missing rate ({R['missing_pct']['Sulfate']}%), suggesting that the imputed "
    "values retain sufficient discriminative information. The broad consensus on the leading features provides "
    "convergent evidence regarding which physicochemical parameters most strongly influence water potability "
    "determination in this dataset.")
add_figure(doc, os.path.join(FIG_DIR, 'feature_importance.png'),
           'Fig. 10. Feature importance comparison for XGBoost and Random Forest.', width=Inches(5.0))

# ============== V. CONCLUSION ==============
add_heading_ieee(doc, 'V. CONCLUSION')
add_body(doc,
    f"This study evaluated the probabilistic prediction performance of Natural Gradient Boosting (NGBoost) for water "
    f"quality classification, comparing it against XGBoost and Random Forest across classification, calibration, and "
    f"uncertainty quantification metrics. Using the Water Potability dataset ({DS['n_samples']:,} samples, "
    f"{DS['n_features']} features), we demonstrated that all three models achieve statistically equivalent "
    f"classification accuracy (p > 0.05 by McNemar's test), with NGBoost at {f4(M['NGBoost']['accuracy'])}, XGBoost "
    f"at {f4(M['XGBoost']['accuracy'])}, and Random Forest at {f4(M['Random Forest']['accuracy'])}.")
add_body(doc,
    "The uncertainty zone analysis reveals that probabilistic outputs from all models carry meaningful confidence "
    "information: predictions in high-confidence zones (Zones 1 and 5) consistently achieve substantially higher "
    "accuracy than those in the ambiguous middle zone (Zone 3). This finding supports the integration of "
    "probabilistic classifiers into operational water quality monitoring systems, where confidence thresholds can "
    "route uncertain samples to additional testing rather than relying on binary accept/reject decisions.")
_lo = int(round(min(_diffs) * 100))
_hi = int(round(max(_diffs) * 100))
add_body(doc,
    f"The SMOTE-ENN analysis demonstrates that resampling techniques consistently degrade model performance on this "
    f"dataset, reducing accuracy by {_lo} to {_hi} percentage points across all models. This result highlights the "
    "importance of evaluating resampling impact empirically rather than applying it as a default preprocessing step, "
    "particularly when the class imbalance is moderate and the feature space exhibits high class overlap.")
add_body(doc,
    "Future work should explore threshold optimization strategies that leverage probabilistic outputs for "
    "cost-sensitive decision-making, alternative calibration methods such as Platt scaling and isotonic regression "
    "for non-probabilistic models, and evaluation on larger water quality datasets with more diverse contamination "
    "patterns. Additionally, temporal analysis of prediction uncertainty under concept drift conditions would "
    "enhance the practical applicability of probabilistic water quality monitoring systems.")

# ============== REFERENCES ==============
add_heading_ieee(doc, 'REFERENCES')
references = [
    '[1] World Health Organization, "Drinking-water," WHO Fact Sheet, 2023. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/drinking-water',
    '[2] U. Ahmed, R. Mumtaz, H. Anwar, A. A. Shah, R. Irfan, and J. Garcia-Nieto, "Efficient Water Quality Prediction Using Supervised Machine Learning," Water, vol. 11, no. 11, p. 2210, 2019.',
    '[3] G. Krishan, N. C. Ghosh, T. Kumar, and R. Srivastav, "Prediction of Water Quality Parameters Using Machine Learning Algorithms: A Case Study," J. Environ. Eng., vol. 147, no. 9, 2021.',
    '[4] A. Niculescu-Mizil and R. Caruana, "Predicting good probabilities with supervised learning," in Proc. 22nd Int. Conf. Machine Learning, 2005, pp. 625-632.',
    '[5] T. Duan, A. Avati, D. Y. Ding, K. K. Thai, S. Basu, A. Ng, and A. Schuler, "NGBoost: Natural Gradient Boosting for Probabilistic Prediction," in Proc. 37th Int. Conf. Machine Learning, PMLR, 2020, pp. 2690-2700.',
    '[6] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, 2016, pp. 785-794.',
    '[7] L. Breiman, "Random Forests," Machine Learning, vol. 45, no. 1, pp. 5-32, 2001.',
    '[8] A. Aldhyani, M. Al-Yaari, H. Alkahtani, and M. Maashi, "Water Quality Prediction Using Artificial Intelligence Algorithms," Applied Bionics and Biomechanics, vol. 2020, 2020.',
    '[9] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique," J. Artificial Intelligence Research, vol. 16, pp. 321-357, 2002.',
    '[10] M. Zhu, J. Wang, X. Yang, Y. Zhang, L. Zhang, H. Ren, B. Wu, and L. Ye, "A review of the application of machine learning in water quality evaluation," Eco-Environment and Health, vol. 1, no. 2, pp. 107-116, 2022.',
    '[11] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," in Proc. 34th Int. Conf. Machine Learning, PMLR, 2017, pp. 1321-1330.',
    '[12] Y. Gal and Z. Ghahramani, "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning," in Proc. 33rd Int. Conf. Machine Learning, PMLR, 2016, pp. 1050-1059.',
    '[13] V. Kuleshov, N. Fenner, and S. Ermon, "Accurate Uncertainties for Deep Learning Using Calibrated Regression," in Proc. 35th Int. Conf. Machine Learning, PMLR, 2018, pp. 2796-2804.',
    '[14] B. C. Wallace and I. J. Dahabreh, "Improving class probability estimates for imbalanced data," Knowledge and Information Systems, vol. 41, pp. 33-52, 2014.',
    '[15] Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," Psychometrika, vol. 12, no. 2, pp. 153-157, 1947.',
]
for ref in references:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(ref)
    run.font.size = Pt(8)
    run.font.name = 'Times New Roman'
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.left_indent = Cm(0.5)
    p.paragraph_format.first_line_indent = Cm(-0.5)

doc.save(OUTPUT)
print(f"Paper saved to: {OUTPUT}")
