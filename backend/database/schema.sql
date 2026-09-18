-- ==============================================================================
-- 🛰️ SatQuery AI — PostgreSQL Enterprise Schema
-- Problem Statement ID: 26167 (ISRO / SAC)
-- Path: backend/database/schema.sql
-- ==============================================================================

-- 1. Multi-turn Chat Sessions
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id VARCHAR(64) PRIMARY KEY,
    title VARCHAR(255) NOT NULL DEFAULT 'New Satellite Dialogue',
    user_id VARCHAR(64) DEFAULT 'commander',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata_json JSONB DEFAULT '{}'::jsonb
);

-- 2. Continuous Chat Messages
CREATE TABLE IF NOT EXISTS chat_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role VARCHAR(16) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    image_id VARCHAR(128),
    metadata_json JSONB DEFAULT '{}'::jsonb,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, id);

-- 3. MemPalace Knowledge Wings (Semantic & Vector Precedents)
CREATE TABLE IF NOT EXISTS mempalace_wings (
    id BIGSERIAL PRIMARY KEY,
    wing_name VARCHAR(64) NOT NULL,
    record_key VARCHAR(128) NOT NULL UNIQUE,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    tags TEXT[],
    embedding_json JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_mempalace_wing ON mempalace_wings(wing_name);
CREATE INDEX IF NOT EXISTS idx_mempalace_tags ON mempalace_wings USING gin(tags);

-- 4. Spatial Areas of Interest (Bounding Box & AOI Geometry)
CREATE TABLE IF NOT EXISTS spatial_aois (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(64) REFERENCES chat_sessions(session_id) ON DELETE SET NULL,
    name VARCHAR(255) NOT NULL,
    crs VARCHAR(64) NOT NULL DEFAULT 'EPSG:4326',
    bbox_wgs84 REAL[],
    geojson JSONB DEFAULT '{}'::jsonb,
    metadata_json JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_spatial_aois_session ON spatial_aois(session_id);

-- 5. Merkle Audit Ledger Records (Tamper-Evident Provenance)
CREATE TABLE IF NOT EXISTS merkle_audit_records (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(64),
    analysis_id VARCHAR(64) UNIQUE,
    merkle_root VARCHAR(128) NOT NULL,
    computation_mode VARCHAR(32) NOT NULL,
    confidence REAL NOT NULL,
    traces_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_merkle_root ON merkle_audit_records(merkle_root);

-- 6. Uploaded Image Metadata (Persistent across server restarts)
CREATE TABLE IF NOT EXISTS uploaded_images (
    image_id VARCHAR(128) PRIMARY KEY,
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
