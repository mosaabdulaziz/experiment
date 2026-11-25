import { useEffect, useRef, useState } from 'react'

interface Staged {
  file: File
  url: string
}

/** Staged multi-photo picker shared by the add-employee form and the
 * existing-employee "إضافة صور" upload widget (2026-09-09). Nothing is
 * uploaded here -- it just holds thumbnails locally so the admin can add to,
 * remove from, and pick one primary among a batch before submitting.
 *
 * Primary is always index 0 of the staged list: clicking a thumbnail moves
 * it to the front instead of tracking a separate "which index is primary"
 * state, so there's no index bookkeeping to get wrong on add/remove. This
 * intentionally matches the backend's own rule (the first file in the
 * upload array that validates becomes the primary photo -- see server.py's
 * admin_upload_photos / register_employee_photo.py) so putting the admin's
 * choice first is the whole fix; no API change needed.
 *
 * Uncontrolled: the parent doesn't own the staged list, only receives it via
 * onChange on every add/remove/reorder. To clear it (e.g. after a
 * successful submit), remount with a different `key` -- the same trick
 * AddEmployeeForm already uses for its other fields via plain useState
 * resets, just applied through React instead. */
export function PhotoPicker({ onChange, disabled }: {
  onChange: (files: File[]) => void
  disabled?: boolean
}) {
  const [staged, setStaged] = useState<Staged[]>([])
  const stagedRef = useRef(staged)
  stagedRef.current = staged

  useEffect(() => {
    onChange(staged.map(s => s.file))
  }, [staged])

  // Revoke every still-live object URL on unmount (picker cleared via
  // remount, or the whole form closed) -- individual removals revoke their
  // own URL immediately, this only catches what's left at teardown.
  useEffect(() => () => {
    for (const s of stagedRef.current) URL.revokeObjectURL(s.url)
  }, [])

  const addFiles = (list: FileList | null) => {
    if (!list || list.length === 0) return
    const added = Array.from(list).map(file => ({ file, url: URL.createObjectURL(file) }))
    setStaged(s => [...s, ...added])
  }

  const removeAt = (i: number) => {
    setStaged(s => {
      URL.revokeObjectURL(s[i].url)
      return s.filter((_, idx) => idx !== i)
    })
  }

  const makePrimary = (i: number) => {
    if (i === 0) return
    setStaged(s => {
      const copy = [...s]
      const [item] = copy.splice(i, 1)
      copy.unshift(item)
      return copy
    })
  }

  return (
    <div className="photoPicker">
      {staged.map((s, i) => (
        <div
          key={s.url}
          className={`photoThumb${i === 0 ? ' photoThumbPrimary' : ''}`}
          onClick={() => makePrimary(i)}
          title={i === 0 ? 'الصورة الأساسية' : 'اضغط لجعلها الصورة الأساسية'}
        >
          <img src={s.url} alt="" />
          {i === 0 && <span className="photoThumbBadge">أساسية</span>}
          <button
            type="button"
            className="photoThumbRemove"
            aria-label="إزالة الصورة"
            disabled={disabled}
            onClick={e => { e.stopPropagation(); removeAt(i) }}
          >
            ×
          </button>
        </div>
      ))}
      <label className={`photoThumbAdd${disabled ? ' disabled' : ''}`}>
        <span aria-hidden="true">⬆</span>
        <input
          type="file" accept="image/*" multiple hidden disabled={disabled}
          onChange={e => { addFiles(e.target.files); e.target.value = '' }}
        />
      </label>
    </div>
  )
}
