"""
Stripe payment integration for international users
"""
import stripe
import os
from typing import Dict

# Stripe configuration
stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "sk_test_your_key")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "whsec_your_secret")


class StripePayment:
    """Stripe payment gateway"""
    
    @staticmethod
    def create_checkout_session(
        user_id: str,
        plan_id: str,
        success_url: str,
        cancel_url: str
    ) -> Dict:
        """
        Create Stripe checkout session
        
        Args:
            user_id: User ID
            plan_id: Pricing plan ID
            success_url: Success redirect URL
            cancel_url: Cancel redirect URL
        
        Returns:
            Checkout session data
        """
        from app.payment.vnpay import get_pricing_plan
        
        plan = get_pricing_plan(plan_id)
        if not plan:
            raise ValueError(f"Invalid plan ID: {plan_id}")
        
        try:
            session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price_data': {
                        'currency': 'usd',
                        'product_data': {
                            'name': plan['name'],
                            'description': plan['description'],
                        },
                        'unit_amount': int(plan['price_usd'] * 100),  # Convert to cents
                    },
                    'quantity': 1,
                }],
                mode='payment',
                success_url=success_url,
                cancel_url=cancel_url,
                client_reference_id=user_id,
                metadata={
                    'user_id': user_id,
                    'plan_id': plan_id,
                    'months': plan['months']
                }
            )
            
            return {
                'session_id': session.id,
                'url': session.url
            }
        
        except stripe.error.StripeError as e:
            raise Exception(f"Stripe error: {str(e)}")
    
    @staticmethod
    def verify_webhook_signature(payload: bytes, sig_header: str) -> Dict:
        """
        Verify Stripe webhook signature
        
        Args:
            payload: Request body
            sig_header: Stripe signature header
        
        Returns:
            Event data
        """
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, STRIPE_WEBHOOK_SECRET
            )
            return event
        except ValueError:
            raise ValueError("Invalid payload")
        except stripe.error.SignatureVerificationError:
            raise ValueError("Invalid signature")
    
    @staticmethod
    def get_payment_intent(payment_intent_id: str) -> Dict:
        """Get payment intent details"""
        try:
            return stripe.PaymentIntent.retrieve(payment_intent_id)
        except stripe.error.StripeError as e:
            raise Exception(f"Stripe error: {str(e)}")
