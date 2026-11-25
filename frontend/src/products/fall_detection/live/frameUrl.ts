/** The camera's latest published frame — the same endpoint the dashboard wall draws
 * (iris/components/streams/StreamTile.tsx). The JPEG already carries the people's boxes
 * painted by the sidecar, so the tab draws the frame and nothing else. */
export const frameUrl = (camera: string): string => `/stream/${encodeURIComponent(camera)}/frame`
