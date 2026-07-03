# Injection Molding Defect Prediction
### Presentation deck and talking points

This is my slide plan for the showcase. Each slide has what goes on the screen, and a "Say" note with what I talk through. The order follows the required flow: the problem and why it matters, the solution, a live demo, and lessons learned.

---

## Slide 1 - Title

On screen:
- Injection Molding Defect Prediction
- Predicting good vs scrap parts from a machine's own settings
- James Mahinda

Say: Hi, I am James. My project predicts whether a plastic molding machine just made a good part or a defective one, using only the machine's own settings, and it does it right away instead of waiting for inspection.

---

## Slide 2 - The problem and why it matters

On screen:
- Photo of the injection molding machine (images/machine_photo.jpg)
- Factories catch defective parts late, during inspection
- By then material and machine time are already wasted
- Scrap costs money on every single cycle

Say: This is an injection molding machine. Almost everything plastic around you is made on one of these. The problem is that when a machine makes a bad part, the factory usually only finds out later during inspection. By then the plastic and the machine time are already wasted, and if a whole batch drifted, that is a lot of scrap. That waste is what I want to catch early.

---

## Slide 3 - How the machine works

On screen:
- Diagram of the mold filling (images/machine_diagram.png)
- Feed, melt, inject, cool, eject
- One part per cycle
- Every cycle the machine records its settings (temperatures, pressures, cycle time)

Say: Quickly, how it works. Plastic pellets go in, a heated screw melts them like melted chocolate, the machine injects that under pressure into a mold, it cools and hardens, then gets ejected. That is one cycle, one part. The key thing for me is that every cycle the machine writes down its own settings. Those settings are my data.

---

## Slide 4 - The idea

On screen:
- Use the machine's recorded settings to predict good vs scrap, immediately
- And point to which settings caused the risk
- So an operator can fix the machine before making more scrap

Say: My idea is simple. Instead of waiting for inspection, use the settings the machine already records to predict scrap on the spot, and also show which settings pushed it toward scrap, so the operator knows what to correct right away.

---

## Slide 5 - The data

On screen:
- Real production data from a plastics company (iGuzzini)
- 1,451 molding cycles
- 13 process parameters per cycle plus a quality label
- Clean: no missing values, no duplicates

Say: I used a real public dataset of 1,451 molding cycles from an actual plastics company, with 13 settings recorded per cycle and a quality label. It is clean industrial data, which let me focus on the modeling rather than fixing broken data.

---

## Slide 6 - Exploring the data (EDA)

On screen:
- Good vs scrap box plots (from eda.ipynb section 7)
- Several settings clearly differ between good and scrap
- Cycle time separates them the most
- So there is real signal to learn from

Say: Before modeling I explored the data. The important question was whether good and scrap parts actually have different settings. They do. Cycle time especially separates them almost cleanly. That told me a model has real signal to work with.

---

## Slide 7 - The smart question: can a straight line separate them?

On screen:
- PCA 2D scatter (from modeling.ipynb section 3)
- Good and scrap overlap, no single straight line splits them
- Straight line model: scrap F1 about 0.74
- Curved model: scrap F1 about 0.85

Say: Here is the part I am most proud of. Before jumping to fancy models, I tested whether a simple straight line could separate good from scrap. I squashed the 13 settings into a 2D picture with PCA, and the two groups overlap, no straight line splits them. I confirmed it with models: a straight line scored 0.74, a curved one jumped to 0.85. So the data genuinely needs a more flexible model. I am not using advanced models to look clever, I am using them because a straight line is not enough.

---

## Slide 8 - The models and results

On screen:
- Model comparison bar chart (scrap precision, recall, F1)
- Logistic Regression (baseline): F1 0.71
- Curved SVM: F1 0.85
- XGBoost: F1 0.97, accuracy 98%
- Chosen model: XGBoost

Say: I trained three models. The simple baseline got 0.71, the curved SVM 0.85, and XGBoost was best by a clear margin at 0.97, about 98 percent accuracy. Out of 74 scrap parts in the test set it caught around 71 and missed only about 3. So XGBoost is my model.

---

## Slide 9 - Explaining the prediction (SHAP)

On screen:
- SHAP summary plot (from modeling.ipynb section 5)
- For any part, SHAP shows which settings pushed it toward scrap
- Turns a black box into something an operator can act on

Say: A prediction alone is not that useful on a factory floor. So I used SHAP, which for any flagged part tells you exactly which settings drove the decision. Instead of just saying scrap, it says this part is scrap because of these two or three settings, look there first. That is what makes it actionable.

---

## Slide 10 - Catching new faults without labels

On screen:
- Isolation Forest, unsupervised anomaly detection
- Learns what a normal cycle looks like, flags the odd ones
- No labels needed
- Catches about a third of scrap on its own

Say: I also added an anomaly detector that needs no labels at all. It learns what a normal cycle looks like and flags the unusual ones. On its own it catches about a third of the scrap. It is a backup for a factory that has no labeled data yet, or a brand new fault the main model was never trained on.

---

## Slide 11 - Being honest about it

On screen:
- The high accuracy partly reflects the dataset
- The quality grades were made at deliberately different machine settings
- In a live factory the signal would be subtler
- The method holds, it would need retraining on that factory's own cycles

Say: I want to be honest about the 98 percent. It is partly because in this dataset the quality grades were produced at deliberately different settings, so the settings line up strongly with the outcome. In a real factory where everything runs at one target setting, the signal would be subtler and the scores lower. The approach still works, it would just need retraining on that factory's own data.

---

## Slide 12 - Live demo

On screen:
- Streamlit app
- Enter a cycle's settings, get back the scrap risk and the top settings behind it

Say: Let me show it working. (Demo: open the app, enter or load a cycle, show the risk score and the top contributing settings, maybe show a clearly good one and a clearly scrap one.)

---

## Slide 13 - Lessons learned

On screen:
- Check the simple thing first (linear separability) before reaching for advanced models
- Explaining a model matters as much as its accuracy
- Being honest about why a score is high builds trust
- Real value is in making the output actionable, not just accurate

Say: A few things I took from this. First, test the simple option before the fancy one, it gives you a real reason for your choices. Second, on a shop floor explaining the model matters as much as the score. And third, being upfront about the limits actually makes the work more credible, not less.

---

## Slide 14 - Next steps and thank you

On screen:
- Drift forecasting: predict when the machine is trending toward scrap (ARIMA)
- Connect to a live machine feed inside a factory system
- Thank you, questions

Say: Next I want to add a forecasting piece that predicts when the machine is drifting toward scrap conditions, before it even happens, and eventually connect it to a live machine feed. Thank you, I am happy to take questions.

---

## Anticipated questions and answers

Q: Why not just use the simple model?
A: Because I tested it. A straight line only reached 0.74 while a curved model reached 0.85, so the data needs the flexibility.

Q: Isn't 98 percent too good to be real?
A: Partly yes, and I say so. The grades were made at different settings so they line up with the outcome. In a live plant it would be lower and would need retraining on that plant's data.

Q: Why XGBoost over the SVM?
A: On the same test set XGBoost scored 0.97 on the scrap class versus 0.85 for the SVM, with fewer missed defects and fewer false alarms.

Q: What if the factory has no labels?
A: That is why I added the Isolation Forest anomaly detector, it needs no labels and still catches a good share of scrap.
