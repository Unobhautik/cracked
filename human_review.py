"""
Human-in-the-Loop Workflow
Routes high-risk cases to human clinicians for review
"""
from typing import Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path


class HumanReviewWorkflow:
    """Manages human review workflow for high-risk medical cases"""
    
    def __init__(self, review_log_path: str = "human_reviews.jsonl"):
        self.review_log_path = Path(review_log_path)
        self.pending_reviews = []
        self.load_pending_reviews()
    
    def load_pending_reviews(self):
        """Load pending reviews from log file"""
        if self.review_log_path.exists():
            try:
                with open(self.review_log_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            review = json.loads(line)
                            if review.get("status") == "pending":
                                self.pending_reviews.append(review)
            except Exception as e:
                print(f"Warning: Could not load review log: {e}")
    
    def create_review_request(self, user_query: str, risk_assessment: Dict, ai_response: str, context: Optional[Dict] = None) -> Dict:
        """Create a review request for human clinician"""
        review_id = f"REV{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        review_request = {
            "review_id": review_id,
            "timestamp": datetime.now().isoformat(),
            "status": "pending",
            "user_query": user_query,
            "risk_level": risk_assessment.get("risk_level", "unknown"),
            "risk_details": {
                "is_emergency": risk_assessment.get("is_emergency", False),
                "is_high_risk": risk_assessment.get("is_high_risk", False),
                "is_medication_risk": risk_assessment.get("is_medication_risk", False),
                "messages": risk_assessment.get("messages", [])
            },
            "ai_response": ai_response,
            "context": context or {},
            "clinician_review": None,
            "clinician_notes": None,
            "approved": False
        }
        
        # Save to log
        self._save_review(review_request)
        self.pending_reviews.append(review_request)
        
        return review_request
    
    def _save_review(self, review: Dict):
        """Save review to log file"""
        try:
            with open(self.review_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(review, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Warning: Could not save review: {e}")
    
    def get_pending_reviews(self) -> List[Dict]:
        """Get all pending reviews"""
        return [r for r in self.pending_reviews if r.get("status") == "pending"]
    
    def submit_review(self, review_id: str, clinician_notes: str, approved: bool, modified_response: Optional[str] = None):
        """Submit clinician review"""
        # Find and update review
        for review in self.pending_reviews:
            if review.get("review_id") == review_id:
                review["status"] = "reviewed"
                review["clinician_notes"] = clinician_notes
                review["approved"] = approved
                review["reviewed_at"] = datetime.now().isoformat()
                if modified_response:
                    review["modified_response"] = modified_response
                
                # Update in log file (in production, use a proper database)
                self._update_review_in_log(review)
                return True
        
        return False
    
    def _update_review_in_log(self, updated_review: Dict):
        """Update review in log file (simplified - in production use proper DB)"""
        # This is a simplified version - in production, use a proper database
        # For now, we'll just append the update
        try:
            with open(self.review_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(updated_review, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Warning: Could not update review: {e}")
    
    def format_review_for_clinician(self, review: Dict) -> str:
        """Format review request for clinician display"""
        return f"""
REVIEW REQUEST: {review['review_id']}
Timestamp: {review['timestamp']}
Risk Level: {review['risk_level'].upper()}

USER QUERY:
{review['user_query']}

RISK ASSESSMENT:
- Emergency: {review['risk_details']['is_emergency']}
- High Risk: {review['risk_details']['is_high_risk']}
- Medication Risk: {review['risk_details']['is_medication_risk']}
- Messages: {', '.join(review['risk_details']['messages'])}

AI RESPONSE:
{review['ai_response']}

CONTEXT:
{json.dumps(review.get('context', {}), indent=2)}
"""
    
    def should_route_to_human(self, risk_assessment: Dict, confidence_score: float = 0.5) -> bool:
        """Determine if case should be routed to human review"""
        # Route if:
        # 1. Emergency situation
        # 2. High risk symptoms
        # 3. Medication risk
        # 4. Low confidence (below threshold)
        
        if risk_assessment.get("is_emergency"):
            return True
        if risk_assessment.get("is_high_risk"):
            return True
        if risk_assessment.get("is_medication_risk"):
            return True
        if confidence_score < 0.5:
            return True
        
        return False


# Global workflow instance
_workflow = None

def get_human_review_workflow() -> HumanReviewWorkflow:
    """Get or create global human review workflow instance"""
    global _workflow
    if _workflow is None:
        _workflow = HumanReviewWorkflow()
    return _workflow

