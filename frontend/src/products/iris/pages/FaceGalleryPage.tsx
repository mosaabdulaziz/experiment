import { useEffect, useState, useCallback, useRef, useMemo } from 'react'
import { facesApi, type FaceEntry, type FaceCapture } from '@/api/faces'
import { IconTrash, IconPlus, IconCamera, IconUpload, IconChevronDown } from '@/components/ui/icons'
import styles from './FaceGalleryPage.module.css'

// ── PersonSection sub-component ────────────────────────────────────────────

function PersonSection({ person, captures, onDeletePerson, onDeleteCapture, onAddImage }: {
  person: FaceEntry
  captures: FaceCapture[]
  onDeletePerson: (id: string) => void
  onDeleteCapture: (id: string) => void
  onAddImage: (personId: string, file: File) => Promise<void>
}) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    onAddImage(person.id, file)
      .finally(() => {
        setUploading(false)
        // Reset input so the same file can be re-selected
        if (fileInputRef.current) fileInputRef.current.value = ''
      })
  }

  return (
    <section className={styles.personSection}>
      <div className={styles.personTop}>
        {/* Main profile photo */}
        <div className={styles.personPhoto}>
          <img
            className={styles.personPhotoImg}
            src={person.photo_path ? `/api/faces/${person.id}/image` : '/face-placeholder.svg'}
            alt={person.name}
            onError={e => {
              (e.currentTarget as HTMLImageElement).src = 'data:image/svg+xml,' +
                encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120"><rect fill="#1a1a24" width="120" height="120"/><text x="60" y="68" text-anchor="middle" fill="#555" font-size="40">👤</text></svg>')
            }}
          />
        </div>

        {/* Capture thumbnails strip + add button */}
        <div className={styles.captureStrip}>
          {captures.map(cap => (
            <div key={cap.id} className={styles.captureTile}>
              <img
                className={styles.captureTileImg}
                src={facesApi.captureImageUrl(cap.id)}
                alt={`Capture ${cap.id}`}
                onError={e => {
                  (e.currentTarget as HTMLImageElement).src = 'data:image/svg+xml,' +
                    encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80"><rect fill="#1a1a24" width="80" height="80"/><text x="40" y="46" text-anchor="middle" fill="#555" font-size="28">❓</text></svg>')
                }}
              />
              <button
                className={styles.captureTileDelete}
                onClick={() => onDeleteCapture(cap.id)}
                title="Remove capture"
              >
                <IconTrash width={10} height={10} />
              </button>
              <span className={styles.captureTileTime}>
                {new Date(cap.captured_at * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
            </div>
          ))}

          {/* Add Image tile */}
          <button
            className={styles.addImageTile}
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            title="Add additional image"
          >
            {uploading ? (
              <span className={styles.addImageTileLabel}>⏳</span>
            ) : (
              <>
                <IconPlus width={18} height={18} />
                <span className={styles.addImageTileLabel}>Add Image</span>
              </>
            )}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={handleFileChange}
          />
        </div>

        {/* Controls column */}
        <div className={styles.personControls}>
          <span className={styles.personName}>{person.name}</span>
          <span className={styles.personDate}>
            Added {new Date(person.created_at * 1000).toLocaleDateString()}
          </span>
          <div className={styles.personActions}>
            <button
              className={styles.deleteBtn}
              onClick={() => onDeletePerson(person.id)}
              title="Delete person"
            >
              <IconTrash width={12} height={12} />
            </button>
          </div>
        </div>
      </div>
    </section>
  )
}

export function FaceGalleryPage() {
  const [known, setKnown] = useState<FaceEntry[]>([])
  const [captures, setCaptures] = useState<FaceCapture[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // ── Assign-capture modal state ─────────────────────────────────────────
  const [assignTarget, setAssignTarget] = useState<FaceCapture | null>(null)
  const [assignName, setAssignName] = useState('')
  const [assigning, setAssigning] = useState(false)

  // ── Add-person modal state ────────────────────────────────────────────
  const [showAddModal, setShowAddModal] = useState(false)
  const [addName, setAddName] = useState('')
  const [addFile, setAddFile] = useState<File | null>(null)
  const [adding, setAdding] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const addPreviewUrl = useMemo(() => addFile ? URL.createObjectURL(addFile) : null, [addFile])

  // Revoke the object URL when the file changes or modal closes
  useEffect(() => {
    return () => {
      if (addPreviewUrl) URL.revokeObjectURL(addPreviewUrl)
    }
  }, [addPreviewUrl])

  const loadData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [knownList, captureList] = await Promise.all([
        facesApi.listKnown(),
        facesApi.listCaptures(),
      ])
      setKnown(knownList)
      setCaptures(captureList)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load face data')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { loadData() }, [loadData])

  const handleDeleteKnown = async (id: string) => {
    try {
      await facesApi.deleteKnown(id)
      setKnown(prev => prev.filter(f => f.id !== id))
    } catch (e) {
      console.error('Failed to delete face:', e)
    }
  }

  const handleDeleteCapture = async (id: string) => {
    try {
      await facesApi.deleteCapture(id)
      setCaptures(prev => prev.filter(c => c.id !== id))
    } catch (e) {
      console.error('Failed to delete capture:', e)
    }
  }

  const handleAddImage = async (personId: string, file: File) => {
    try {
      await facesApi.addImage(personId, file)
      await loadData()
    } catch (e) {
      console.error('Failed to add image:', e)
    }
  }

  const handleAssign = async () => {
    if (!assignTarget || !assignName.trim()) return
    setAssigning(true)
    try {
      await facesApi.assignCapture(assignTarget.id, assignName.trim())
      await loadData()
      setAssignTarget(null)
      setAssignName('')
    } catch (e) {
      console.error('Failed to assign capture:', e)
    } finally {
      setAssigning(false)
    }
  }

  const handleAddPerson = async () => {
    if (!addName.trim() || !addFile) return
    setAdding(true)
    try {
      await facesApi.register(addName.trim(), addFile)
      await loadData()
      setShowAddModal(false)
      setAddName('')
      setAddFile(null)
    } catch (e) {
      console.error('Failed to add person:', e)
    } finally {
      setAdding(false)
    }
  }

  // ── Group captures by assigned person name ─────────────────────────────
  const assignedCaptures = useMemo(() => {
    const map: Record<string, FaceCapture[]> = {}
    for (const cap of captures) {
      if (cap.assigned_to) {
        (map[cap.assigned_to] ??= []).push(cap)
      }
    }
    return map
  }, [captures])

  const unassignedCaptures = useMemo(
    () => captures.filter(c => !c.assigned_to),
    [captures],
  )

  const totalCaptureCount = captures.length

  return (
    <div className={styles.page}>
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <header className={styles.header}>
        <div className={styles.headLeft}>
          <h1 className={styles.title}>Face Gallery</h1>
          <p className={styles.subtitle}>
            {known.length} person{known.length !== 1 ? 's' : ''} ·{' '}
            {totalCaptureCount} capture{totalCaptureCount !== 1 ? 's' : ''}
          </p>
        </div>
        <div className={styles.headerActions}>
          <button className={styles.addPersonBtn} onClick={() => setShowAddModal(true)}>
            <IconPlus width={12} height={12} /> Add Person
          </button>
          <button className={styles.refreshBtn} onClick={loadData} disabled={loading}>
            ⟳ {loading ? 'Loading...' : 'Refresh'}
          </button>
        </div>
      </header>

      {/* ── Error ───────────────────────────────────────────────────────── */}
      {error && <div className={styles.error}>{error}</div>}

      {/* ── Person sections ─────────────────────────────────────────────── */}
      {known.map(person => {
        const personCaptures = assignedCaptures[person.name] ?? []
        return (
          <PersonSection
            key={person.id}
            person={person}
            captures={personCaptures}
            onDeletePerson={handleDeleteKnown}
            onDeleteCapture={handleDeleteCapture}
            onAddImage={handleAddImage}
          />
        )
      })}

      {/* ── Pending / unassigned captures ──────────────────────────────── */}
      {unassignedCaptures.length > 0 && (
        <>
          <div className={styles.sectionDivider}>
            <IconCamera width={13} height={13} />
            <span>Pending Captures ({unassignedCaptures.length})</span>
          </div>
          <div className={styles.pendingGrid}>
            {unassignedCaptures.map(cap => (
              <div key={cap.id} className={styles.card}>
                <div className={styles.cardImgWrap}>
                  <img
                    className={styles.cardImg}
                    src={facesApi.captureImageUrl(cap.id)}
                    alt={`Capture from ${cap.source_id}`}
                    onError={e => {
                      (e.currentTarget as HTMLImageElement).src = 'data:image/svg+xml,' +
                        encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120"><rect fill="#1a1a24" width="120" height="120"/><text x="60" y="68" text-anchor="middle" fill="#555" font-size="40">❓</text></svg>')
                    }}
                  />
                </div>
                <div className={styles.cardBody}>
                  <span className={styles.cardNameUnknown}>Unknown</span>
                  <span className={styles.cardMeta}>
                    {cap.source_id} · {new Date(cap.captured_at * 1000).toLocaleTimeString()}
                  </span>
                </div>
                <div className={styles.cardActions}>
                  <button
                    className={styles.assignBtn}
                    onClick={() => setAssignTarget(cap)}
                    title="Assign to person"
                  >
                    <IconPlus width={12} height={12} /> Assign
                  </button>
                  <button
                    className={styles.deleteBtn}
                    onClick={() => handleDeleteCapture(cap.id)}
                    title="Reject capture"
                  >
                    <IconTrash width={12} height={12} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {/* ── Empty state ─────────────────────────────────────────────────── */}
      {!loading && known.length === 0 && unassignedCaptures.length === 0 && (
        <div className={styles.empty}>
          <p>No faces yet.</p>
          <p className={styles.emptyHint}>
            Add a person to get started, or wait for CCTV captures to appear.
          </p>
        </div>
      )}

      {/* ── Assign modal ────────────────────────────────────────────────── */}
      {assignTarget && (
        <div className={styles.modalBackdrop} onClick={() => setAssignTarget(null)}>
          <div className={styles.modal} onClick={e => e.stopPropagation()}>
            <h3 className={styles.modalTitle}>Assign Capture to Person</h3>
            <img
              className={styles.modalImg}
              src={facesApi.captureImageUrl(assignTarget.id)}
              alt="Face capture"
              onError={e => {
                (e.currentTarget as HTMLImageElement).src = 'data:image/svg+xml,' +
                  encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="160" height="160"><rect fill="#1a1a24" width="160" height="160"/><text x="80" y="88" text-anchor="middle" fill="#555" font-size="50">❓</text></svg>')
              }}
            />
            <p className={styles.modalMeta}>
              From <strong>{assignTarget.source_id}</strong> ·{' '}
              {new Date(assignTarget.captured_at * 1000).toLocaleString()}
            </p>
            <div className={styles.modalSelectWrap}>
              <select
                className={styles.modalSelect}
                value={assignName}
                onChange={e => setAssignName(e.target.value)}
                autoFocus
              >
                <option value="">Select a person…</option>
                {known.map(face => (
                  <option key={face.id} value={face.name}>
                    {face.name}
                  </option>
                ))}
              </select>
              <IconChevronDown width={12} height={12} className={styles.modalSelectChevron} />
            </div>
            <div className={styles.modalActions}>
              <button
                className={styles.modalCancel}
                onClick={() => setAssignTarget(null)}
              >
                Cancel
              </button>
              <button
                className={styles.modalConfirm}
                onClick={handleAssign}
                disabled={!assignName.trim() || assigning}
              >
                {assigning ? 'Assigning...' : 'Assign & Add to Gallery'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Add-person modal ─────────────────────────────────────────────── */}
      {showAddModal && (
        <div className={styles.modalBackdrop} onClick={() => { setShowAddModal(false); setAddFile(null); setAddName('') }}>
          <div className={styles.modal} onClick={e => e.stopPropagation()}>
            <h3 className={styles.modalTitle}>Add New Person</h3>

            <div
              className={styles.addPhotoArea}
              onClick={() => fileInputRef.current?.click()}
            >
              {addFile && addPreviewUrl ? (
                <img
                  className={styles.addPhotoPreview}
                  src={addPreviewUrl}
                  alt="Preview"
                />
              ) : (
                <div className={styles.addPhotoPlaceholder}>
                  <IconUpload width={24} height={24} />
                  <span>Click to upload photo</span>
                </div>
              )}
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              style={{ display: 'none' }}
              onChange={e => {
                const file = e.target.files?.[0]
                if (file) setAddFile(file)
              }}
            />

            <input
              className={styles.modalInput}
              type="text"
              placeholder="Person's name (e.g. John Doe)"
              value={addName}
              onChange={e => setAddName(e.target.value)}
              autoFocus
              onKeyDown={e => {
                if (e.key === 'Enter' && addName.trim() && addFile) handleAddPerson()
                if (e.key === 'Escape') { setShowAddModal(false); setAddFile(null); setAddName('') }
              }}
            />

            <div className={styles.modalActions}>
              <button
                className={styles.modalCancel}
                onClick={() => { setShowAddModal(false); setAddFile(null); setAddName('') }}
              >
                Cancel
              </button>
              <button
                className={styles.modalConfirm}
                onClick={handleAddPerson}
                disabled={!addName.trim() || !addFile || adding}
              >
                {adding ? 'Adding...' : 'Add to Gallery'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
