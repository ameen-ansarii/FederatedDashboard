from app.models.domain import User, UserRole, Job, JobStatus, Dataset


def test_user_creation(db):
    user = User(
        email="test_researcher@fl.org",
        hashed_password="hashed_password_example",
        full_name="Dr. Jane Doe",
        role=UserRole.RESEARCHER,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    assert user.id is not None
    assert user.role == UserRole.RESEARCHER
    assert user.is_active is True


def test_job_creation(db):
    user = User(
        email="job_owner@fl.org",
        hashed_password="password_hash",
        role=UserRole.ADMIN,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    job = Job(
        id="job-uuid-12345",
        type="PREPROCESSING",
        status=JobStatus.PENDING,
        progress=0.0,
        user_id=user.id,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    assert job.id == "job-uuid-12345"
    assert job.status == JobStatus.PENDING
    assert job.user.email == "job_owner@fl.org"
