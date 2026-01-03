"""
Script to update existing FREE users from 5 to 3 notes/day limit
Run this after updating the code
"""
from app.database.database import SessionLocal
from app.database.models import User, AccountType

def update_free_users_limit():
    """Update all FREE users to have 3 notes/day limit"""
    db = SessionLocal()
    try:
        # Find all FREE users with old limit (5)
        free_users = db.query(User).filter(
            User.account_type == AccountType.FREE,
            User.daily_note_limit == 5
        ).all()
        
        count = 0
        for user in free_users:
            user.daily_note_limit = 3
            count += 1
        
        db.commit()
        
        print(f"✅ Updated {count} FREE users to 3 notes/day limit")
        
        # Verify
        remaining = db.query(User).filter(
            User.account_type == AccountType.FREE,
            User.daily_note_limit == 5
        ).count()
        
        if remaining > 0:
            print(f"⚠️  Warning: {remaining} FREE users still have 5 notes/day")
        else:
            print("✅ All FREE users now have 3 notes/day limit")
        
        # Show summary
        free_count = db.query(User).filter(User.account_type == AccountType.FREE).count()
        pro_count = db.query(User).filter(User.account_type == AccountType.PRO).count()
        
        print(f"\n📊 Summary:")
        print(f"   FREE users: {free_count} (3 notes/day)")
        print(f"   PRO users: {pro_count} (unlimited)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("=" * 60)
    print("Updating FREE users rate limit: 5 → 3 notes/day")
    print("=" * 60)
    print()
    
    update_free_users_limit()
    
    print()
    print("=" * 60)
    print("Update complete!")
    print("=" * 60)
