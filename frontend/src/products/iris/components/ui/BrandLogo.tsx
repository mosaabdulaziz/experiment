interface BrandLogoProps {
  /** Rendered edge length in px (the mark is square). */
  size?: number
}

/** The IRIS app icon — a coral glyph on a black squircle tile. Used as the brand mark in the
 *  header, the landing hero, and the chat welcome (and mirrored in public/favicon.svg). */
export function BrandLogo({ size = 32 }: BrandLogoProps) {
  return (
    <svg
      viewBox="0 0 1024 1024"
      width={size}
      height={size}
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <rect y="0" width="1024" height="1024" rx="287.31" ry="287.31" fill="#020302" />
      <g fill="#f9996c">
        <path d="M339.87,516.59v-159.49l-136.81,136.81c-9.99,9.99-9.99,26.19,0,36.18l82.55,82.55h150.32l-96.05-96.05Z" />
        <path d="M714.26,387.16l-75.82-75.82c-4.8-4.8-11.31-7.49-18.09-7.49h-41.5l-83.31,83.31h218.72Z" />
        <path d="M620.35,720.15c6.79,0,13.29-2.69,18.09-7.49l21.04-21.04v-148.92l-177.45,177.45h138.32Z" />
        <path d="M310.44,637.73l74.93,74.93c4.8,4.8,11.31,7.49,18.09,7.49h43.03l82.42-82.42h-218.47Z" />
        <path d="M403.71,303.85c-6.79,0-13.29,2.7-18.09,7.49l-21.04,21.04v150.83l179.36-179.36h-140.23Z" />
        <path d="M820.95,493.85l-.06-.06-81.91-81.91h-149.81l95.03,95.03v160l136.69-136.69v-.13c10.05-9.98,10.08-26.23.06-36.24Z" />
        <path d="M530.96,481.9c-.72-.72-1.69-1.13-2.71-1.13h-32.49c-1.02,0-2,.4-2.72,1.13l-27.38,27.39c-1.5,1.5-1.5,3.93,0,5.43l27.35,27.39c.72.72,1.69,1.12,2.71,1.12h32.53c1.02,0,1.99-.4,2.71-1.12l27.37-27.37v-.02c1.51-1.5,1.51-3.93,0-5.44h0s-27.37-27.37-27.37-27.37Z" />
      </g>
    </svg>
  )
}
