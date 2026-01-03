"""
Payment routes for subscription management
Supports CASSO (Vietnamese users - QR Code) and Stripe (International users)
"""
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Request, Header
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import uuid

from app.database.database import get_db
from app.database.models import User, Payment, AccountType
from app.auth.security import get_current_active_user
from app.payment.casso import CassoService, get_pricing_plan, get_all_pricing_plans
from app.payment.stripe_payment import StripePayment

router = APIRouter(prefix="/payment", tags=["Payment"])


class CreatePaymentRequest(BaseModel):
    """Request to create payment"""
    plan_id: str
    payment_method: str  # casso or stripe


class PaymentResponse(BaseModel):
    """Payment response"""
    payment_method: str
    payment_id: str
    amount: int
    currency: str
    # For CASSO
    qr_code_url: Optional[str] = None
    bank_info: Optional[dict] = None
    transfer_content: Optional[str] = None
    instructions: Optional[list] = None
    expires_at: Optional[str] = None
    status_check_url: Optional[str] = None
    # For Stripe
    checkout_url: Optional[str] = None
    session_id: Optional[str] = None


@router.get("/plans")
async def get_pricing_plans():
    """
    Get all available pricing plans
    Returns plans for both CASSO (VND - QR Code) and Stripe (USD)
    """
    plans = get_all_pricing_plans()
    return {
        "plans": [
            {
                **plan,
                "payment_methods": ["casso", "stripe"],
                "recommended": plan["id"] == "pro_3_months"
            }
            for plan in plans
        ],
        "currency": {
            "vnd": "Vietnamese Dong (CASSO - QR Code)",
            "usd": "US Dollar (Stripe - Card)"
        }
    }


@router.post("/create")
async def create_payment(
    payment_request: CreatePaymentRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create payment for Pro subscription
    
    - **plan_id**: Pricing plan (pro_1_month, pro_3_months, pro_12_months)
    - **payment_method**: Payment gateway (casso for QR code or stripe for card)
    
    Returns:
        - CASSO: QR code URL and bank transfer info
        - Stripe: Checkout URL
    """
    # Get plan details
    plan = get_pricing_plan(payment_request.plan_id)
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gói thanh toán không hợp lệ"
        )
    
    # Check if user already has active subscription
    if current_user.account_type == AccountType.PRO:
        if current_user.subscription_end and current_user.subscription_end > datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Bạn đã có gói PRO đang hoạt động"
            )
    
    # Generate unique payment ID
    payment_id = str(uuid.uuid4())
    
    if payment_request.payment_method == 'casso':
        # CASSO payment with QR code
        amount = plan['price_vnd']
        
        # Create payment record
        payment = Payment(
            id=uuid.UUID(payment_id),
            user_id=current_user.id,
            amount=amount,
            currency='VND',
            payment_method='casso',
            transaction_id=payment_id,
            payment_status='pending',
            plan_type=payment_request.plan_id,
            plan_duration_months=plan['months'],
            metadata={'plan_name': plan['name']}
        )
        
        db.add(payment)
        db.commit()
        
        # Generate CASSO payment with QR code
        casso = CassoService()
        payment_info = casso.create_payment_request(
            order_id=payment_id,
            amount=amount,
            description=f"NotallyX {plan['name']}",
            user_email=current_user.email
        )
        
        return {
            "payment_method": "casso",
            "payment_id": payment_id,
            "qr_code_url": payment_info["qr_code_url"],
            "bank_info": payment_info["bank_info"],
            "transfer_content": payment_info["transfer_content"],
            "amount": amount,
            "currency": "VND",
            "instructions": payment_info["instructions"],
            "expires_at": payment_info["expires_at"],
            "status_check_url": f"/api/payment/status/{payment_id}"
        }
    
    elif payment_request.payment_method == 'stripe':
        # Stripe payment
        amount = plan['price_usd']
        
        # Create payment record
        payment = Payment(
            id=uuid.UUID(payment_id),
            user_id=current_user.id,
            amount=amount,
            currency='USD',
            payment_method='stripe',
            transaction_id=payment_id,
            payment_status='pending',
            plan_type=payment_request.plan_id,
            plan_duration_months=plan['months'],
            metadata={'plan_name': plan['name']}
        )
        
        db.add(payment)
        db.commit()
        
        # Create Stripe checkout session
        stripe_payment = StripePayment()
        session = stripe_payment.create_checkout_session(
            user_id=str(current_user.id),
            plan_id=payment_request.plan_id,
            success_url=f"http://localhost:8000/api/payment/stripe/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"http://localhost:8000/api/payment/stripe/cancel"
        )
        
        # Update transaction ID with Stripe session ID
        payment.transaction_id = session['session_id']
        payment.metadata = {
            **payment.metadata,
            'stripe_session_id': session['session_id']
        }
        db.commit()
        
        return {
            "payment_method": "stripe",
            "payment_id": payment_id,
            "checkout_url": session['url'],
            "session_id": session['session_id'],
            "amount": amount,
            "currency": "USD"
        }
    
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Phương thức thanh toán không hợp lệ. Chọn 'casso' hoặc 'stripe'"
        )


@router.get("/status/{payment_id}")
async def check_payment_status(
    payment_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Check payment status (for CASSO payments)
    Client should poll this endpoint every 5-10 seconds
    """
    # Get payment record
    try:
        payment = db.query(Payment).filter(
            Payment.id == uuid.UUID(payment_id),
            Payment.user_id == current_user.id
        ).first()
    except:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy thanh toán"
        )
    
    if not payment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy thanh toán"
        )
    
    # If already completed, return success
    if payment.payment_status == 'completed':
        return {
            "status": "completed",
            "message": "Thanh toán thành công!",
            "payment_id": payment_id,
            "subscription_end": payment.subscription_end.isoformat() if payment.subscription_end else None
        }
    
    # If CASSO payment, check with CASSO API
    if payment.payment_method == 'casso':
        casso = CassoService()
        
        # Get plan to verify amount
        plan = get_pricing_plan(payment.plan_type)
        expected_amount = plan['price_vnd']
        
        # Verify transaction
        transaction = casso.verify_transaction(
            order_id=payment_id,
            expected_amount=expected_amount,
            time_window_minutes=30
        )
        
        if transaction:
            # Payment found! Upgrade user
            payment.payment_status = 'completed'
            payment.transaction_id = transaction['transaction_id']
            payment.metadata = {
                **payment.metadata,
                'casso_transaction': transaction
            }
            
            # Upgrade user to Pro
            current_user.account_type = AccountType.PRO
            current_user.daily_note_limit = -1  # Unlimited
            current_user.subscription_start = datetime.utcnow()
            current_user.subscription_end = datetime.utcnow() + timedelta(days=30 * payment.plan_duration_months)
            
            payment.subscription_start = current_user.subscription_start
            payment.subscription_end = current_user.subscription_end
            
            db.commit()
            
            return {
                "status": "completed",
                "message": "Thanh toán thành công! Tài khoản của bạn đã được nâng cấp lên Pro.",
                "payment_id": payment_id,
                "subscription_end": current_user.subscription_end.isoformat()
            }
    
    # Still pending
    return {
        "status": "pending",
        "message": "Đang chờ thanh toán...",
        "payment_id": payment_id
    }


@router.post("/casso/webhook")
async def casso_webhook(
    request: Request,
    x_signature: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    CASSO webhook endpoint
    CASSO will call this when a transaction is detected
    """
    # Get raw body for signature verification
    body = await request.body()
    body_str = body.decode('utf-8')
    
    # Verify signature
    casso = CassoService()
    if x_signature:
        is_valid = casso.verify_webhook_signature(body_str, x_signature)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid webhook signature"
            )
    
    # Parse webhook data
    import json
    webhook_data = json.loads(body_str)
    
    # Process webhook
    transaction_info = casso.process_webhook(webhook_data)
    
    if not transaction_info['order_id']:
        return {"status": "ignored", "message": "No order ID found"}
    
    # Find payment record
    try:
        payment = db.query(Payment).filter(
            Payment.id == uuid.UUID(transaction_info['order_id'])
        ).first()
    except:
        return {"status": "ignored", "message": "Invalid order ID"}
    
    if not payment:
        return {"status": "ignored", "message": "Payment not found"}
    
    if payment.payment_status == 'completed':
        return {"status": "already_processed", "message": "Payment already completed"}
    
    # Verify amount
    plan = get_pricing_plan(payment.plan_type)
    if transaction_info['amount'] != plan['price_vnd']:
        return {"status": "amount_mismatch", "message": "Amount does not match"}
    
    # Update payment status
    payment.payment_status = 'completed'
    payment.transaction_id = transaction_info['transaction_id']
    payment.metadata = {
        **payment.metadata,
        'casso_transaction': transaction_info
    }
    
    # Upgrade user to Pro
    user = db.query(User).filter(User.id == payment.user_id).first()
    if user:
        user.account_type = AccountType.PRO
        user.daily_note_limit = -1  # Unlimited
        user.subscription_start = datetime.utcnow()
        user.subscription_end = datetime.utcnow() + timedelta(days=30 * payment.plan_duration_months)
        
        payment.subscription_start = user.subscription_start
        payment.subscription_end = user.subscription_end
    
    db.commit()
    
    return {
        "status": "success",
        "message": "Payment processed successfully",
        "order_id": transaction_info['order_id']
    }


@router.get("/stripe/success")
async def stripe_success(session_id: str, db: Session = Depends(get_db)):
    """Stripe payment success callback"""
    # Get payment record by Stripe session ID
    payment = db.query(Payment).filter(Payment.transaction_id == session_id).first()
    
    if not payment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payment not found"
        )
    
    # Update payment status
    payment.payment_status = 'completed'
    
    # Upgrade user to Pro
    user = db.query(User).filter(User.id == payment.user_id).first()
    if user:
        user.account_type = AccountType.PRO
        user.daily_note_limit = -1  # Unlimited
        user.subscription_start = datetime.utcnow()
        user.subscription_end = datetime.utcnow() + timedelta(days=30 * payment.plan_duration_months)
        
        payment.subscription_start = user.subscription_start
        payment.subscription_end = user.subscription_end
    
    db.commit()
    
    return {
        "status": "success",
        "message": "Payment successful! Your account has been upgraded to Pro.",
        "subscription_end": user.subscription_end.isoformat() if user else None
    }


@router.get("/stripe/cancel")
async def stripe_cancel():
    """Stripe payment cancel callback"""
    return {
        "status": "cancelled",
        "message": "Payment was cancelled"
    }


@router.get("/history")
async def get_payment_history(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user's payment history"""
    payments = db.query(Payment).filter(
        Payment.user_id == current_user.id
    ).order_by(Payment.created_at.desc()).all()
    
    return {
        "payments": [payment.to_dict() for payment in payments]
    }


@router.get("/subscription/status")
async def get_subscription_status(
    current_user: User = Depends(get_current_active_user)
):
    """Get current subscription status"""
    is_pro = current_user.account_type == AccountType.PRO
    
    if is_pro and current_user.subscription_end:
        days_remaining = (current_user.subscription_end - datetime.utcnow()).days
    else:
        days_remaining = 0
    
    return {
        "account_type": current_user.account_type.value,
        "is_pro": is_pro,
        "subscription_start": current_user.subscription_start.isoformat() if current_user.subscription_start else None,
        "subscription_end": current_user.subscription_end.isoformat() if current_user.subscription_end else None,
        "days_remaining": max(0, days_remaining),
        "daily_note_limit": current_user.daily_note_limit,
        "notes_created_today": current_user.notes_created_today
    }
