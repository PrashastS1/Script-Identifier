"use client"

import type React from "react"

import { useState, useRef } from "react"
import Image from "next/image"
import Link from "next/link"
import { motion, AnimatePresence } from "framer-motion"
import { Upload, X, Github, Database } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import FeatureExtractionAnimation from "./feature-extraction-animation"

const languages = [
  "Hindi",
  "Bengali",
  "Telugu",
  "Tamil",
  "Kannada",
  "Malayalam",
  "Gujarati",
  "Punjabi",
  "Urdu",
  "Sanskrit",
  "English",
]

export default function ScriptIdentifier() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [identifiedLanguage, setIdentifiedLanguage] = useState<string | null>(null)
  const [confidenceScore, setConfidenceScore] = useState<number | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleImageUpload(e.dataTransfer.files[0])
    }
  }

  const handleImageUpload = (file: File) => {
    // Reset states
    setError(null)
    setIdentifiedLanguage(null)
    setConfidenceScore(null)
    
    // Validate file is an image
    if (!file.type.startsWith('image/')) {
      setError("Please upload an image file")
      return
    }
    
    const reader = new FileReader()
    reader.onload = (e) => {
      if (e.target?.result) {
        setSelectedImage(e.target.result as string)
        setSelectedFile(file)
      }
    }
    reader.readAsDataURL(file)
  }

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleImageUpload(e.target.files[0])
    }
  }

  const handleClear = () => {
    setSelectedImage(null)
    setSelectedFile(null)
    setIdentifiedLanguage(null)
    setConfidenceScore(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  const handleSubmit = async () => {
    if (!selectedFile) return

    setIsProcessing(true)
    setError(null)

    try {
      // Create a FormData object to send the image
      const formData = new FormData()
      formData.append('image', selectedFile)

      // Send the image to the API endpoint
      const response = await fetch('http://127.0.0.1:8002/pipeline/', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}: ${response.statusText}`)
      }

      const data = await response.json()
      
      // Update state with the response data
      setIdentifiedLanguage(data.language)
      setConfidenceScore(data.confidence_score)
    } catch (err) {
      console.error('Error processing image:', err)
      setError(err instanceof Error ? err.message : 'Failed to process image')
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      <header className="text-center mb-10">
        <motion.h1
          className="text-4xl md:text-5xl font-bold mb-2"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          LinguaLens
        </motion.h1>
        <motion.p
          className="text-lg text-slate-300"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          Developed by Tera bhai Seedhe Maut
        </motion.p>

        <div className="flex justify-between items-center mt-6">
          <div className="w-32 h-32 relative">
            <Image
              src="/IITJ_logo.png"
              alt="IIT Logo"
              width={128}
              height={128}
              className="object-cover rounded"
            />
          </div>

          <div className="flex gap-4 mt-4">
            <Link href="https://github.com/AurindumBanerjee/Script-Identifier" className="text-blue-400 hover:text-blue-300 transition flex items-center gap-2">
              <Github size={18} />
              <span>GitHub Repository</span>
            </Link>
            <Link href="https://drive.google.com/drive/folders/1gjdmyTR_9B7U1-W7hWugewnSowjetXYC?usp=drive_link" className="text-blue-400 hover:text-blue-300 transition flex items-center gap-2">
              <Database size={18} />
              <span>Dataset Repository</span>
            </Link>
          </div>

          <div className="w-32 h-32 relative">
            <Image
              src="/ui_logo.png"
              alt="Profile"
              width={128}
              height={128}
              className="object-cover rounded"
            />
          </div>
        </div>
      </header>

      <div className="grid md:grid-cols-2 gap-8">
        <div className="flex flex-col">
          <div
            className={`relative h-80 border-2 border-dashed rounded-lg flex flex-col items-center justify-center cursor-pointer transition-all duration-200 ${
              isDragging ? "border-blue-500 bg-blue-500/10" : "border-slate-600 hover:border-slate-500"
            } ${selectedImage ? "bg-slate-800/50" : "bg-slate-800/20"}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              accept="image/*"
              onChange={handleFileInputChange}
            />

            {selectedImage ? (
              <div className="relative w-full h-full">
                <Image
                  src={selectedImage}
                  alt="Uploaded script"
                  fill
                  className="object-contain p-2"
                />
                <button
                  className="absolute top-2 right-2 bg-slate-900/80 p-1 rounded-full hover:bg-slate-800"
                  onClick={(e) => {
                    e.stopPropagation()
                    handleClear()
                  }}
                >
                  <X size={16} />
                </button>
              </div>
            ) : (
              <div className="flex flex-col items-center text-slate-400">
                <Upload className="w-10 h-10 mb-2" />
                <p className="text-lg font-medium">Drop Image Here</p>
                <p className="text-sm">- or -</p>
                <p className="text-sm">Click to Upload</p>
              </div>
            )}
          </div>

          {error && (
            <div className="mt-2 p-2 bg-red-500/20 border border-red-500/50 rounded text-red-300 text-sm">
              {error}
            </div>
          )}

          <div className="flex items-center gap-2 mt-4">
            <Button variant="outline" className="flex-1" onClick={handleClear} disabled={!selectedImage}>
              Clear
            </Button>
            <Button
              className="flex-1 bg-blue-600 hover:bg-blue-700"
              onClick={handleSubmit}
              disabled={!selectedImage || isProcessing}
            >
              {isProcessing ? "Processing..." : "Submit"}
            </Button>
          </div>
        </div>

        <div className="flex flex-col">
          <div className="mb-4">
            <h2 className="text-xl font-semibold mb-2">Identified Language</h2>

            <AnimatePresence>
              {isProcessing ? (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="h-80 bg-slate-800/20 rounded-lg border border-slate-700 flex items-center justify-center"
                >
                  <FeatureExtractionAnimation />
                </motion.div>
              ) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="relative"
                >
                  {identifiedLanguage ? (
                    <div className="h-80 bg-slate-800/20 rounded-lg border border-slate-700 flex flex-col items-center justify-center p-6">
                      <motion.div
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ type: "spring", stiffness: 300, damping: 20 }}
                        className="text-4xl font-bold text-center mb-4 text-blue-400"
                      >
                        {identifiedLanguage}
                      </motion.div>

                      <div className="w-full max-w-md">
                        <div className="bg-slate-700/50 rounded-lg p-4 mb-4">
                          <h3 className="text-lg font-medium mb-2">Confidence Score</h3>
                          <div className="w-full bg-slate-800 rounded-full h-4">
                            <motion.div
                              className="bg-blue-600 h-4 rounded-full"
                              initial={{ width: 0 }}
                              animate={{ width: `${confidenceScore || 0}%` }}
                              transition={{ duration: 1, delay: 0.2 }}
                            />
                          </div>
                          <div className="flex justify-between mt-1 text-sm">
                            <span>0%</span>
                            <span className="font-medium">{confidenceScore?.toFixed(1) || 0}%</span>
                            <span>100%</span>
                          </div>
                        </div>

                        <div className="mt-4 text-sm text-slate-400 text-center">
                          <p>
                            {confidenceScore && confidenceScore > 75
                              ? "Analysis complete with high confidence"
                              : confidenceScore && confidenceScore > 50
                              ? "Analysis complete with moderate confidence"
                              : "Analysis complete with low confidence"}
                          </p>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="h-80 bg-slate-800/20 rounded-lg border border-slate-700 flex items-center justify-center text-slate-400">
                      <p>Results will appear here</p>
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  )
}