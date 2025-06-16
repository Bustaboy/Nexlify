// frontend/src/components/common/ErrorBoundary.tsx

import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Home, FileText } from 'lucide-react';
import { motion } from 'framer-motion';

// The safety net - catching crashes before they flatline your trades
// Every error caught here is a catastrophe avoided, a fortune saved
// I've seen this screen more times than I'd like to admit, but less than I could have

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  errorCount: number;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorCount: 0
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    // Update state so the next render will show the fallback UI
    return { 
      hasError: true,
      error
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log error to error reporting service
    console.error('ErrorBoundary caught:', error, errorInfo);
    
    // Update state with error details
    this.setState(prevState => ({
      errorInfo,
      errorCount: prevState.errorCount + 1
    }));

    // Send to error tracking (if available)
    if (window.nexlify) {
      window.nexlify.fs.writeFile(
        `logs/crash_reports/crash_${Date.now()}.json`,
        JSON.stringify({
          timestamp: new Date().toISOString(),
          error: {
            message: error.message,
            stack: error.stack
          },
          errorInfo: {
            componentStack: errorInfo.componentStack
          },
          userAgent: navigator.userAgent,
          url: window.location.href
        }, null, 2)
      ).catch(console.error);
    }
  }

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null
    });
  };

  handleReload = () => {
    window.location.reload();
  };

  handleGoHome = () => {
    window.location.href = '/';
  };

  handleCopyError = () => {
    const errorText = `
Error: ${this.state.error?.message}
Stack: ${this.state.error?.stack}
Component Stack: ${this.state.errorInfo?.componentStack}
    `.trim();

    navigator.clipboard.writeText(errorText);
    
    // Show toast or feedback
    const button = document.getElementById('copy-error-btn');
    if (button) {
      button.textContent = 'Copied!';
      setTimeout(() => {
        button.textContent = 'Copy Error Details';
      }, 2000);
    }
  };

  render() {
    if (this.state.hasError) {
      // Custom fallback UI
      if (this.props.fallback) {
        return <>{this.props.fallback}</>;
      }

      // Default error UI - styled for Night City
      return (
        <div className="min-h-screen bg-cyber-black flex items-center justify-center p-4">
          {/* Background effects */}
          <div className="absolute inset-0 opacity-5">
            <div className="cyber-grid" />
          </div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
            className="relative max-w-2xl w-full"
          >
            <div className="bg-cyber-dark border border-neon-red/50 rounded-lg p-8 shadow-2xl">
              {/* Error header */}
              <div className="flex items-center space-x-4 mb-6">
                <div className="relative">
                  <AlertTriangle className="w-12 h-12 text-neon-red" />
                  <div className="absolute inset-0 blur-sm">
                    <AlertTriangle className="w-12 h-12 text-neon-red" />
                  </div>
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-white">
                    System Malfunction
                  </h1>
                  <p className="text-gray-400 text-sm mt-1">
                    The neural matrix hit a snag. Don't worry, your funds are safe.
                  </p>
                </div>
              </div>

              {/* Error details */}
              <div className="bg-cyber-black rounded-lg p-4 mb-6 font-mono text-sm">
                <div className="text-neon-red mb-2">
                  {this.state.error?.name}: {this.state.error?.message}
                </div>
                
                {/* Show stack trace in dev mode */}
                {process.env.NODE_ENV === 'development' && (
                  <details className="mt-4">
                    <summary className="cursor-pointer text-gray-500 hover:text-gray-300">
                      Technical Details (click to expand)
                    </summary>
                    <pre className="mt-2 text-xs text-gray-600 overflow-auto max-h-40">
                      {this.state.error?.stack}
                    </pre>
                    {this.state.errorInfo && (
                      <pre className="mt-2 text-xs text-gray-600 overflow-auto max-h-40">
                        {this.state.errorInfo.componentStack}
                      </pre>
                    )}
                  </details>
                )}
              </div>

              {/* Error count warning */}
              {this.state.errorCount > 2 && (
                <div className="bg-neon-yellow/10 border border-neon-yellow/50 rounded-lg p-4 mb-6">
                  <p className="text-neon-yellow text-sm">
                    Multiple errors detected. The system might be unstable.
                    Consider refreshing the page or checking your connection.
                  </p>
                </div>
              )}

              {/* Recovery actions */}
              <div className="grid grid-cols-2 gap-4">
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={this.handleReset}
                  className="flex items-center justify-center space-x-2 px-4 py-3 bg-cyber-black border border-neon-cyan/50 rounded-lg text-neon-cyan hover:bg-neon-cyan/10 transition-colors"
                >
                  <RefreshCw className="w-4 h-4" />
                  <span>Try Again</span>
                </motion.button>

                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={this.handleReload}
                  className="flex items-center justify-center space-x-2 px-4 py-3 bg-cyber-black border border-gray-600 rounded-lg text-gray-300 hover:bg-gray-900 transition-colors"
                >
                  <RefreshCw className="w-4 h-4" />
                  <span>Reload Page</span>
                </motion.button>

                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={this.handleGoHome}
                  className="flex items-center justify-center space-x-2 px-4 py-3 bg-cyber-black border border-gray-600 rounded-lg text-gray-300 hover:bg-gray-900 transition-colors"
                >
                  <Home className="w-4 h-4" />
                  <span>Go Home</span>
                </motion.button>

                <motion.button
                  id="copy-error-btn"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={this.handleCopyError}
                  className="flex items-center justify-center space-x-2 px-4 py-3 bg-cyber-black border border-gray-600 rounded-lg text-gray-300 hover:bg-gray-900 transition-colors"
                >
                  <FileText className="w-4 h-4" />
                  <span>Copy Error Details</span>
                </motion.button>
              </div>

              {/* Helpful tips */}
              <div className="mt-6 text-xs text-gray-500">
                <p className="mb-2">Common fixes that might help:</p>
                <ul className="list-disc list-inside space-y-1">
                  <li>Check your internet connection</li>
                  <li>Clear your browser cache</li>
                  <li>Disable browser extensions temporarily</li>
                  <li>Try a different browser</li>
                  {this.state.errorCount > 1 && (
                    <li className="text-neon-yellow">
                      Consider restarting the application
                    </li>
                  )}
                </ul>
              </div>

              {/* Emergency contact */}
              <div className="mt-6 pt-6 border-t border-cyber-gray text-center">
                <p className="text-xs text-gray-600">
                  If problems persist, your trades are safe.
                  Error ID: {Date.now().toString(36).toUpperCase()}
                </p>
              </div>
            </div>

            {/* Decorative elements */}
            <div className="absolute -top-4 -left-4 w-24 h-24 bg-neon-red/20 rounded-full blur-xl" />
            <div className="absolute -bottom-4 -right-4 w-32 h-32 bg-neon-red/10 rounded-full blur-2xl" />
          </motion.div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Hook for functional components to handle errors
export const useErrorHandler = () => {
  return (error: Error, errorInfo?: ErrorInfo) => {
    console.error('Error caught by hook:', error, errorInfo);
    
    // You could also trigger a state update or send to error tracking
    throw error; // Re-throw to be caught by ErrorBoundary
  };
};

// Wrapper component for async errors
export const AsyncErrorBoundary: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [error, setError] = React.useState<Error | null>(null);

  React.useEffect(() => {
    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      setError(new Error(event.reason));
    };

    window.addEventListener('unhandledrejection', handleUnhandledRejection);

    return () => {
      window.removeEventListener('unhandledrejection', handleUnhandledRejection);
    };
  }, []);

  if (error) {
    throw error;
  }

  return <>{children}</>;
};
