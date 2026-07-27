from pymongo import MongoClient
from app.core.config import MONGO_URI, MONGO_DB_NAME, MONGO_COLLECTION_NAME
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MongoDatabase:

    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self._connected = False
        self.connect()

    def connect(self):
        try:
            # Configure a public DNS resolver to bypass local router DNS issues with SRV records
            try:
                import dns.resolver
                dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
                dns.resolver.default_resolver.nameservers = ['8.8.8.8', '8.8.4.4', '1.1.1.1']
            except Exception as dns_err:
                logger.warning(f"Could not configure public DNS resolver: {dns_err}")

            logger.info(f"Connecting to MongoDB at URI: {MONGO_URI.split('@')[-1] if '@' in MONGO_URI else MONGO_URI}")
            self.client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            # Trigger server selection to verify connection
            self.client.admin.command('ping')
            self.db = self.client[MONGO_DB_NAME]
            self.collection = self.db[MONGO_COLLECTION_NAME]
            
            # Ensure index on chunk_id for quick lookups
            self.collection.create_index("chunk_id", unique=True)
            self._connected = True
            logger.info("Successfully connected to MongoDB!")
        except Exception as e:
            self._connected = False
            err_msg = str(e)
            if "TLSV1_ALERT_INTERNAL_ERROR" in err_msg or "SSL handshake failed" in err_msg:
                logger.warning(
                    "\n" + "="*80 + "\n"
                    "MONGODB CONNECTION WARNING: SSL Handshake Failed.\n"
                    "This usually means your current IP address is not whitelisted in MongoDB Atlas.\n"
                    "Please log in to your MongoDB Atlas dashboard, go to 'Network Access' under Security,\n"
                    "and ensure your IP address is whitelisted (or add 0.0.0.0/0 to allow access from anywhere for testing).\n"
                    "="*80 + "\n"
                )
            logger.warning(f"Failed to connect to MongoDB: {e}. Falling back to local pickle files.")

    @property
    def is_connected(self) -> bool:
        if not self._connected:
            return False
        try:
            self.client.admin.command('ping')
            return True
        except Exception:
            self._connected = False
            return False

    def save_chunks(self, chunks: list):
        """
        Saves or updates chunk metadata documents in the database.
        Prevents inserting duplicate data if the identical dataset already exists.
        """
        if not self.is_connected:
            logger.warning("MongoDB not connected. Skipping save.")
            return False

        try:
            # 1. Check if the identical dataset already exists to prevent duplicate entry
            existing_count = self.collection.count_documents({})
            if existing_count == len(chunks) and len(chunks) > 0:
                first_match = self.collection.find_one({"chunk_id": 0, "text": chunks[0]["text"]})
                mid_idx = len(chunks) // 2
                mid_match = self.collection.find_one({"chunk_id": mid_idx, "text": chunks[mid_idx]["text"]})
                last_match = self.collection.find_one({"chunk_id": len(chunks) - 1, "text": chunks[-1]["text"]})
                
                if first_match and mid_match and last_match:
                    logger.info("Identical dataset already exists in MongoDB. Skipping save to prevent duplicate entries.")
                    return True

            # 2. If it's a new or modified dataset, clear old records to avoid collisions
            self.collection.delete_many({})
            
            # 3. Insert new chunks
            if chunks:
                # Remove '_id' if present and ensure chunk_id matches list index
                for idx, c in enumerate(chunks):
                    c.pop('_id', None)
                    c["chunk_id"] = idx
                self.collection.insert_many(chunks)
            logger.info(f"Successfully saved {len(chunks)} chunks to MongoDB.")
            return True
        except Exception as e:
            logger.error(f"Error saving chunks to MongoDB: {e}")
            return False

    def get_chunks_by_ids(self, chunk_ids: list) -> dict:
        """
        Fetches chunk documents matching the given chunk_ids.
        Returns a dict mapping chunk_id to chunk document.
        """
        if not self.is_connected:
            logger.warning("MongoDB not connected. Cannot fetch chunks.")
            return {}

        try:
            cursor = self.collection.find({"chunk_id": {"$in": chunk_ids}})
            return {doc["chunk_id"]: doc for doc in cursor}
        except Exception as e:
            logger.error(f"Error fetching chunks from MongoDB: {e}")
            return {}
