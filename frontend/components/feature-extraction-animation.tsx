"use client"

import { useEffect, useRef } from "react"
import { motion } from "framer-motion"

export default function FeatureExtractionAnimation() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    // Set canvas dimensions
    canvas.width = canvas.offsetWidth
    canvas.height = canvas.offsetHeight
    
    // Create a simulated image
    const imageWidth = canvas.width * 0.6
    const imageHeight = canvas.height * 0.6
    const imageX = (canvas.width - imageWidth) / 2
    const imageY = (canvas.height - imageHeight) / 2
    
    // Feature points
    const numPoints = 50
    const points: {x: number, y: number, size: number, active: boolean}[] = []
    
    for (let i = 0; i < numPoints; i++) {
      points.push({
        x: imageX + Math.random() * imageWidth,
        y: imageY + Math.random() * imageHeight,
        size: 1 + Math.random() * 3,
        active: false
      })
    }
    
    // Lines connecting points to classification
    const lines: {start: {x: number, y: number}, end: {x: number, y: number}, progress: number, speed: number}[] = []
    
    // Classification result position
    const resultX = canvas.width * 0.8
    const resultY = canvas.height * 0.5
    
    // Animation loop
    let animationFrame: number
    let lastActivationTime = 0
    let activePointIndex = 0
    
    const animate = (timestamp: number) => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      
      // Draw simulated image outline
      ctx.strokeStyle = 'rgba(100, 150, 255, 0.3)'
      ctx.strokeRect(imageX, imageY, imageWidth, imageHeight)
      
      // Activate points over time
      if (timestamp - lastActivationTime > 100 && activePointIndex < points.length) {
        points[activePointIndex].active = true
        
        // Create a line from this point to the result
        lines.push({
          start: { x: points[activePointIndex].x, y: points[activePointIndex].y },
          end: { x: resultX, y: resultY },
          progress: 0,
          speed: 0.01 + Math.random() * 0.03
        })
        
        lastActivationTime = timestamp
        activePointIndex++
      }
      
      // Draw and update points
      points.forEach(point => {
        ctx.beginPath()
        ctx.arc(point.x, point.y, point.size, 0, Math.PI * 2)
        
        if (point.active) {
          ctx.fillStyle = 'rgba(100, 150, 255, 0.8)'
        } else {
          ctx.fillStyle = 'rgba(100, 150, 255, 0.2)'
        }
        
        ctx.fill()
      })
      
      // Draw and update lines
      lines.forEach((line, index) => {
        line.progress += line.speed
        line.progress = Math.min(line.progress, 1)
        
        const currentX = line.start.x + (line.end.x - line.start.x) * line.progress
        const currentY = line.start.y + (line.end.y - line.start.y) * line.progress
        
        ctx.beginPath()
        ctx.moveTo(line.start.x, line.start.y)
        ctx.lineTo(currentX, currentY)
        ctx.strokeStyle = `rgba(100, 150, 255, ${0.1 + line.progress * 0.4})`
        ctx.lineWidth = 1
        ctx.stroke()
        
        // Draw pulse at the end of the line
        if (line.progress > 0.05) {
          ctx.beginPath()
          ctx.arc(currentX, currentY, 2, 0, Math.PI * 2)
          ctx.fillStyle = 'rgba(100, 150, 255, 0.8)'
          ctx.fill()
        }
      })
      
      // Draw classification result
      ctx.beginPath()
      ctx.arc(resultX, resultY, 15, 0, Math.PI * 2)
      ctx.fillStyle = 'rgba(100, 150, 255, 0.2)'
      ctx.fill()
      
      ctx.beginPath()
      ctx.arc(resultX, resultY, 12, 0, Math.PI * 2)
      ctx.fillStyle = 'rgba(100, 150, 255, 0.3)'
      ctx.fill()
      
      // Count completed lines
      const completedLines = lines.filter(line => line.progress >= 1).length
      
      // Draw progress in the center
      if (points.length > 0 && lines.length > 0) {
        const progress = completedLines / points.length
        ctx.font = '12px Arial'
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)'
        ctx.textAlign = 'center'
        ctx.fillText(`${Math.round(progress * 100)}%`, resultX, resultY + 4)
      }
      
      animationFrame = requestAnimationFrame(animate)
    }
    
    animationFrame = requestAnimationFrame(animate)
    
    return () => {
      cancelAnimationFrame(animationFrame)
    }
  }, [])
  
  return (
    <div className="w-full h-full flex flex-col items-center justify-center">
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="text-center mb-4"
      >
        <h3 className="text-lg font-medium text-blue-400 mb-1">Analyzing Script Features</h3>
        <p className="text-sm text-slate-400">Extracting visual patterns and characteristics</p>
      </motion.div>
      
      <canvas 
        ref={canvasRef} 
        className="w-full h-[70%]"
      />
    </div>
  )
}
