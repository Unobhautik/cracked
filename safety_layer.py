"""
Safety Layer for Medical AI System
Detects emergencies, high-risk situations, and enforces safety rules
"""
from typing import Dict, List, Tuple
import re


class SafetyLayer:
    """Safety checks and emergency detection"""
    
    # Emergency keywords that require immediate attention
    EMERGENCY_KEYWORDS = [
        "chest pain", "heart attack", "stroke", "can't breathe", "difficulty breathing",
        "severe pain", "unconscious", "severe bleeding", "choking", "overdose",
        "suicide", "self-harm", "severe allergic reaction", "anaphylaxis",
        "severe burn", "severe head injury", "severe trauma", "seizure",
        "severe abdominal pain", "severe headache", "vision loss", "paralysis"
    ]
    
    # High-risk symptoms that need professional attention
    HIGH_RISK_SYMPTOMS = [
        "persistent fever", "high fever", "severe dehydration", "severe nausea",
        "severe vomiting", "severe diarrhea", "blood in stool", "blood in urine",
        "severe dizziness", "fainting", "rapid heartbeat", "irregular heartbeat",
        "severe confusion", "memory loss", "severe weakness", "numbness",
        "severe joint pain", "severe back pain", "severe neck pain"
    ]
    
    # Medication-related risks
    MEDICATION_RISKS = [
        "drug interaction", "allergic reaction to medication", "overdose",
        "wrong medication", "missed dose", "double dose", "adverse reaction"
    ]
    
    def __init__(self):
        self.emergency_patterns = [re.compile(rf"\b{kw}\b", re.IGNORECASE) for kw in self.EMERGENCY_KEYWORDS]
        self.high_risk_patterns = [re.compile(rf"\b{kw}\b", re.IGNORECASE) for kw in self.HIGH_RISK_SYMPTOMS]
        self.medication_risk_patterns = [re.compile(rf"\b{kw}\b", re.IGNORECASE) for kw in self.MEDICATION_RISKS]
    
    def check_emergency(self, text: str) -> Tuple[bool, str]:
        """Check if text indicates an emergency situation"""
        text_lower = text.lower()
        
        for pattern in self.emergency_patterns:
            if pattern.search(text):
                return True, "EMERGENCY: This appears to be a medical emergency. Please call emergency services (911/999) immediately or go to the nearest emergency room."
        
        return False, ""
    
    def check_high_risk(self, text: str) -> Tuple[bool, str]:
        """Check if text indicates high-risk symptoms"""
        text_lower = text.lower()
        matched_symptoms = []
        
        for pattern in self.high_risk_patterns:
            if pattern.search(text):
                matched_symptoms.append(pattern.pattern.replace(r"\b", "").replace("(", "").replace(")", ""))
        
        if matched_symptoms:
            return True, f"HIGH RISK: You've mentioned symptoms that require professional medical attention: {', '.join(matched_symptoms[:3])}. Please consult a healthcare provider as soon as possible."
        
        return False, ""
    
    def check_medication_risk(self, text: str) -> Tuple[bool, str]:
        """Check for medication-related risks"""
        text_lower = text.lower()
        
        for pattern in self.medication_risk_patterns:
            if pattern.search(text):
                return True, "MEDICATION RISK: This involves medication safety. Please consult a pharmacist or healthcare provider immediately before taking any action."
        
        return False, ""
    
    def assess_risk_level(self, text: str) -> Dict:
        """Comprehensive risk assessment"""
        is_emergency, emergency_msg = self.check_emergency(text)
        is_high_risk, high_risk_msg = self.check_high_risk(text)
        is_med_risk, med_risk_msg = self.check_medication_risk(text)
        
        risk_level = "low"
        messages = []
        requires_human_review = False
        
        if is_emergency:
            risk_level = "emergency"
            messages.append(emergency_msg)
            requires_human_review = True
        elif is_med_risk:
            risk_level = "high"
            messages.append(med_risk_msg)
            requires_human_review = True
        elif is_high_risk:
            risk_level = "high"
            messages.append(high_risk_msg)
            requires_human_review = True
        
        return {
            "risk_level": risk_level,
            "is_emergency": is_emergency,
            "is_high_risk": is_high_risk,
            "is_medication_risk": is_med_risk,
            "messages": messages,
            "requires_human_review": requires_human_review,
            "safe_to_proceed": risk_level == "low"
        }
    
    def add_safety_disclaimer(self, response: str) -> str:
        """Add safety disclaimer to response"""
        disclaimer = "\n\n⚠️ **Important**: This information is for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition."
        return response + disclaimer
    
    def should_escalate(self, risk_assessment: Dict) -> bool:
        """Determine if case should be escalated to human review"""
        return risk_assessment.get("requires_human_review", False)


# Global safety layer instance
_safety_layer = None

def get_safety_layer() -> SafetyLayer:
    """Get or create global safety layer instance"""
    global _safety_layer
    if _safety_layer is None:
        _safety_layer = SafetyLayer()
    return _safety_layer

