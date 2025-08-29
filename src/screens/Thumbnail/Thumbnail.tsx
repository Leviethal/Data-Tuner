import {
  CopyIcon,
  FileTextIcon,
  MessageSquareIcon,
  MoreHorizontalIcon,
  SendIcon,
  SettingsIcon,
  SparklesIcon,
  UploadIcon,
} from "lucide-react";
import React from "react";
import { Avatar, AvatarFallback } from "../../components/ui/avatar";
import { Button } from "../../components/ui/button";
import { Card, CardContent } from "../../components/ui/card";
import { Input } from "../../components/ui/input";
import { ScrollArea } from "../../components/ui/scroll-area";

export const Thumbnail = (): JSX.Element => {
  const [csvFile, setCsvFile] = React.useState<File | null>(null);
  const [prompt, setPrompt] = React.useState("");

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type === "text/csv") {
      setCsvFile(file);
    } else {
      alert("Please select a valid CSV file");
    }
  };

  const handleSubmit = async () => {
    if (!csvFile) {
      alert("Please upload a CSV file first");
      return;
    }
    if (!prompt.trim()) {
      alert("Please enter a prompt");
      return;
    }

    const formData = new FormData();
    formData.append('file', csvFile);
    formData.append('prompt', prompt);

    try {
      const response = await fetch('http://localhost:5000/api/upload', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const errorData = await response.json();
        alert(errorData.error || 'Upload failed');
        return;
      }
      await response.json();
      alert('File uploaded and processed successfully!');
      // Optionally handle the response data here
    } catch (error) {
      alert('An error occurred while uploading the file.');
    }

    setCsvFile(null);
    setPrompt("");
    // Reset file input
    const fileInput = document.getElementById('csv-upload') as HTMLInputElement;
    if (fileInput) fileInput.value = '';
  };

  const conversations: any[] = []; // Empty conversations array

  const suggestedPrompts = [
    "Clean and optimize this dataset",
    "Remove outliers and handle missing values",
    "Apply feature scaling and normalization",
    "Detect and fix data quality issues",
  ];

  return (
    <div className="bg-gradient-to-br from-[#e8f0fe] to-[#f3e8ff] min-h-screen w-full flex items-center justify-center p-4">
      <Card className="w-full max-w-6xl h-[800px] bg-white/90 backdrop-blur-sm shadow-2xl rounded-3xl overflow-hidden">
        <CardContent className="p-0 h-full flex">
          {/* Sidebar */}
          <div className="w-80 bg-white/50 backdrop-blur-sm border-r border-gray-200/50 flex flex-col">
            {/* Header */}
            <div className="p-6 border-b border-gray-200/50">
              <div className="flex items-center justify-between mb-4">
                <h1 className="text-xl font-bold text-gray-800">Data Optimise</h1>
                <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                  <MoreHorizontalIcon className="h-4 w-4" />
                </Button>
              </div>
              
              {/* CSV Upload Section */}
              <div className="space-y-3">
                <label htmlFor="csv-upload" className="block">
                  <div className="w-full border-2 border-dashed border-gray-300 rounded-lg p-4 text-center cursor-pointer hover:border-blue-400 hover:bg-blue-50/50 transition-colors">
                    <UploadIcon className="h-6 w-6 mx-auto mb-2 text-gray-400" />
                    <p className="text-sm text-gray-600">
                      {csvFile ? csvFile.name : "Upload raw dataset (CSV)"}
                    </p>
                    <p className="text-xs text-gray-400 mt-1">
                      Raw data for ML preprocessing
                    </p>
                  </div>
                </label>
                <input
                  id="csv-upload"
                  type="file"
                  accept=".csv"
                  onChange={handleFileChange}
                  className="hidden"
                />
                
                {csvFile && (
                  <div className="flex items-center p-2 bg-green-50 rounded-lg">
                    <FileTextIcon className="h-4 w-4 text-green-600 mr-2" />
                    <span className="text-sm text-green-700 truncate">
                      {csvFile.name}
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Conversations or Empty State */}
            <div className="flex-1 p-4">
              {conversations.length > 0 ? (
                <>
                  <div className="flex items-center justify-between mb-4">
                    <span className="text-sm text-gray-600">
                      Your conversations
                    </span>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-blue-500 text-sm h-auto p-0"
                    >
                      Clear All
                    </Button>
                  </div>

                  <ScrollArea className="h-full">
                    <div className="space-y-2">
                      {conversations.map((conversation) => (
                        <div
                          key={conversation.id}
                          className={`flex items-center p-3 rounded-lg cursor-pointer transition-colors ${
                            conversation.active
                              ? "bg-blue-50 border border-blue-200"
                              : "hover:bg-gray-50"
                          }`}
                        >
                          <MessageSquareIcon className="h-4 w-4 mr-3 text-gray-500" />
                          <span className="text-sm text-gray-700 truncate">
                            {conversation.title}
                          </span>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </>
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-center">
                  <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mb-4">
                    <MessageSquareIcon className="h-8 w-8 text-gray-400" />
                  </div>
                  <p className="text-sm text-gray-500 mb-2">Ready to get Started?</p>
                  <p className="text-xs text-gray-400">
                    Start a chat to begin
                  </p>
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="p-4 border-t border-gray-200/50">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center">
                  <SettingsIcon className="h-4 w-4 mr-2 text-gray-500" />
                  <span className="text-sm text-gray-700">Settings</span>
                </div>
              </div>
              <div className="flex items-center">
                <Avatar className="h-8 w-8 mr-3">
                  <AvatarFallback className="bg-gray-200 text-gray-600 text-sm">
                    AN
                  </AvatarFallback>
                </Avatar>
                <span className="text-sm text-gray-700">Andrew NG</span>
              </div>
            </div>
          </div>

          {/* Main Chat Area */}
          <div className="flex-1 flex flex-col">
            {/* Welcome Screen */}
            <div className="flex-1 flex flex-col items-center justify-center p-6">
              <div className="max-w-2xl mx-auto text-center">
                {/* AI Logo */}
                <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center mb-8 mx-auto">
                  <SparklesIcon className="h-10 w-10 text-white" />
                </div>

                {/* Welcome Message */}
                <h1 className="text-4xl font-bold text-gray-800 mb-4">
                  Automated Data Cleaning & Optimization
                </h1>
                <p className="text-lg text-gray-600 mb-12">
                  Upload raw datasets and specify Cleaning requirements. Our algorithms will automatically apply the best preprocessing techniques for machine learning.
                </p>

                {/* Suggested Prompts */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-12">
                  {suggestedPrompts.map((prompt, index) => (
                    <Button
                      key={index}
                      variant="outline"
                      onClick={() => setPrompt(prompt)}
                      className="p-4 h-auto text-left justify-start bg-white/50 hover:bg-white/80 border-gray-200 rounded-xl transition-all duration-200 hover:shadow-md"
                    >
                      <div className="flex items-start">
                        <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0" />
                        <span className="text-sm text-gray-700">{prompt}</span>
                      </div>
                    </Button>
                  ))}
                </div>

                {/* Features */}
                <div className="grid grid-cols-3 gap-6 text-center">
                  <div className="space-y-2">
                    <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto">
                      <FileTextIcon className="h-6 w-6 text-blue-600" />
                    </div>
                    <h3 className="font-medium text-gray-800">Data Analysis</h3>
                    <p className="text-xs text-gray-500">
                      Automated outlier detection & removal
                    </p>
                  </div>
                  <div className="space-y-2">
                    <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto">
                      <SparklesIcon className="h-6 w-6 text-green-600" />
                    </div>
                    <h3 className="font-medium text-gray-800">ML Optimization</h3>
                    <p className="text-xs text-gray-500">
                      Feature scaling & normalization
                    </p>
                  </div>
                  <div className="space-y-2">
                    <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mx-auto">
                      <CopyIcon className="h-6 w-6 text-purple-600" />
                    </div>
                    <h3 className="font-medium text-gray-800">Smart Processing</h3>
                    <p className="text-xs text-gray-500">
                      Best algorithm selection automatically
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Input Area */}
            <div className="p-6 border-t border-gray-200/50">
              <div className="max-w-4xl mx-auto">
                <div className="relative">
                  <Input
                    placeholder="Describe your data cleaning requirements (e.g., 'remove outliers, handle missing values, must use StandardScaler')..."
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    className="w-full pr-12 py-4 rounded-full border-gray-300 focus:border-blue-500 focus:ring-blue-500 bg-white/80 backdrop-blur-sm text-base"
                  />
                  <Button
                    onClick={handleSubmit}
                    size="sm"
                    className={`absolute right-2 top-1/2 transform -translate-y-1/2 h-10 w-10 p-0 rounded-full transition-colors ${csvFile && prompt.trim() ? 'bg-blue-500 hover:bg-blue-600' : 'bg-gray-300 cursor-not-allowed'}`}
                    disabled={!csvFile || !prompt.trim()}
                  >
                    <SendIcon className="h-4 w-4" />
                  </Button>
                </div>
                <p className="text-xs text-gray-500 text-center mt-3">
                  Specify cleaning requirements and optional algorithm preferences. Processed data is returned immediately - nothing is stored.
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};