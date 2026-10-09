from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.database import get_db
from app.core.errors import UnauthorizedException, ForbiddenException
from app.models.domain import User, UserRole

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/token",
    auto_error=False
)


def get_current_user(
    db: Session = Depends(get_db), token: Optional[str] = Depends(oauth2_scheme)
) -> Optional[User]:
    if not token:
        # For open/development access or anonymous endpoints where user is optional
        return None
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise UnauthorizedException(detail="Could not validate token subject")
    except Exception:
        raise UnauthorizedException(detail="Invalid authentication token")

    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user:
        raise UnauthorizedException(detail="User not found")
    if not user.is_active:
        raise ForbiddenException(detail="Inactive user")
    return user


def require_role(allowed_roles: list[UserRole]):
    def role_checker(current_user: Optional[User] = Depends(get_current_user)) -> User:
        if not current_user:
            raise UnauthorizedException(detail="Authentication required")
        if current_user.role not in allowed_roles and current_user.role != UserRole.ADMIN:
            raise ForbiddenException(detail="Insufficient permissions for this resource")
        return current_user

    return role_checker
