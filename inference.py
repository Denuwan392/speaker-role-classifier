from feature_pipeline import RoleFeatureExtractor
from agentic_router import HybridRoleRouter

def main():
    print("Initializing components...")
    
    # 1. Initialize Feature Extractor
    extractor = RoleFeatureExtractor(model_dir="models")
    
    # 2. Initialize the Hybrid Router
    router = HybridRoleRouter(model_dir="models", threshold=0.85)
    
    # 3. Mock dictionary of test strings
    test_cases = {"Manager Profile": """
Let's start the sprint review.

Can everyone provide an update?

Please finish the API work by Thursday.

Escalate blockers immediately.

We'll review progress on Friday.
""",
"HR Profile": """
Please complete the mandatory harassment prevention training by Friday.

Open enrollment for employee benefits begins next week.

The HR portal now contains the updated grievance reporting procedures.

Remember that mental health leave requests can be submitted separately from annual PTO.

If you need assistance navigating the employee benefits system, please contact HR.
"""
}
    
    print("\nStarting inference...\n" + "-"*40)
    
    # Fetch feature names explicitly from the router to guarantee alignment with XGBoost
    feature_names = router.feature_names
    
    # 4. Loop through strings and run inference
    for case_name, text in test_cases.items():
        print(f"Case: {case_name}")
        print(f"Text: \"{text}\"")
        
        # Extract deterministic features
        features = extractor.transform(text, feature_names=feature_names)
        
        # Route logic (predict_proba -> fallback check)
        probs = router.xgb_model.predict_proba(features)[0]

        print("\nProbability Distribution:")
        for role, prob in zip(router.label_encoder.classes_, probs):
            print(f"  {role:<10} : {prob:.3f}")

        predicted_role, source, confidence = router.predict(features, text)

        print(f"\nPredicted Role : {predicted_role}")
        print(f"Deciding System: {source}")
        print(f"Confidence     : {confidence:.2f} (from XGBoost)")
        print("-" * 40)

        print(features.shape)
        print(features)

if __name__ == "__main__":
    main()
