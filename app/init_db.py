from app.database import engine, Base
from app import models  # noqa: F401 - imported so SQLAlchemy registers the tables

def init_database():
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Done! Database file 'smarthire.db' created with all tables.")

if __name__ == "__main__":
    init_database()