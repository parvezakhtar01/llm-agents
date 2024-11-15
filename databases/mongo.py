# database/mongodb.py
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Dict, Any
import json

class MongoDB:
    def __init__(self, mongo_uri: str, database: str):
        self.client = AsyncIOMotorClient(mongo_uri)
        self.db = self.client[database]
        
    async def save_job_result(self, job_id: str, data: Dict[str, Any]) -> bool:
        """Save job result to MongoDB"""
        try:
            collection = self.db.job_results
            await collection.update_one(
                {"job_id": job_id},
                {"$set": data},
                upsert=True
            )
            return True
        except Exception as e:
            print(f"Error saving to MongoDB: {str(e)}")
            return False

    async def get_job_result(self, job_id: str) -> Dict[str, Any]:
        """Retrieve job result from MongoDB"""
        try:
            collection = self.db.job_results
            result = await collection.find_one({"job_id": job_id})
            return result if result else None
        except Exception as e:
            print(f"Error retrieving from MongoDB: {str(e)}")
            return None
