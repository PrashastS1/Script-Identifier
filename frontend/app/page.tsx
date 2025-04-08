import ScriptIdentifier from "@/components/script-identifier"
import { ThemeProvider } from "@/components/theme-provider"

export default function Home() {
  return (
    <ThemeProvider defaultTheme="dark">
      <main className="min-h-screen bg-gradient-to-b from-slate-950 to-slate-900 text-white">
        <ScriptIdentifier />
      </main>
    </ThemeProvider>
  )
}
