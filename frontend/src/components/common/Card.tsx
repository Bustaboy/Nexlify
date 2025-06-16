// frontend/src/components/common/Card.tsx

import React from 'react';
import { motion } from 'framer-motion';
import { cn } from '@lib/utils';
import { LucideIcon } from 'lucide-react';

// The container that holds our data - like a memory chip in the matrix
// Each card a window into different data streams, bordered by neon, filled with purpose
// I've stared at thousands of these, each one holding fortunes won and lost

interface CardProps {
  children: React.ReactNode;
  className?: string;
  title?: string;
  subtitle?: string;
  icon?: LucideIcon;
  action?: React.ReactNode;
  variant?: 'default' | 'glass' | 'neon' | 'danger' | 'success';
  noPadding?: boolean;
  interactive?: boolean;
  onClick?: () => void;
  glowColor?: string;
  animate?: boolean;
}

export const Card: React.FC<CardProps> = ({
  children,
  className,
  title,
  subtitle,
  icon: Icon,
  action,
  variant = 'default',
  noPadding = false,
  interactive = false,
  onClick,
  glowColor,
  animate = true
}) => {
  // Variant styles - each tells a different story
  const variantStyles = {
    default: 'bg-cyber-black border-cyber-dark',
    glass: 'bg-glass-light backdrop-blur-md border-glass-medium',
    neon: 'bg-cyber-black border-neon-cyan shadow-neon-cyan',
    danger: 'bg-cyber-black border-neon-red shadow-neon-red',
    success: 'bg-cyber-black border-neon-green shadow-neon-green'
  };

  const cardContent = (
    <>
      {/* Header - where the story begins */}
      {(title || action) && (
        <div className={cn(
          "flex items-center justify-between",
          noPadding ? "p-4" : "mb-4",
          "border-b border-cyber-dark"
        )}>
          <div className="flex items-center space-x-3">
            {Icon && (
              <div className="relative">
                <Icon className={cn(
                  "w-5 h-5",
                  variant === 'neon' && "text-neon-cyan",
                  variant === 'danger' && "text-neon-red",
                  variant === 'success' && "text-neon-green"
                )} />
                {/* Icon glow effect */}
                {(variant === 'neon' || variant === 'danger' || variant === 'success') && (
                  <div className="absolute inset-0 blur-sm opacity-50">
                    <Icon className={cn(
                      "w-5 h-5",
                      variant === 'neon' && "text-neon-cyan",
                      variant === 'danger' && "text-neon-red",
                      variant === 'success' && "text-neon-green"
                    )} />
                  </div>
                )}
              </div>
            )}
            <div>
              <h3 className="text-sm font-medium text-white">{title}</h3>
              {subtitle && (
                <p className="text-xs text-gray-500 mt-0.5">{subtitle}</p>
              )}
            </div>
          </div>
          {action && (
            <div className="flex items-center">{action}</div>
          )}
        </div>
      )}

      {/* Content - where the data lives */}
      <div className={cn(!noPadding && (title ? "px-4 pb-4" : "p-4"))}>
        {children}
      </div>

      {/* Scan line effect - that CRT nostalgia */}
      {variant === 'default' && (
        <div className="absolute inset-0 pointer-events-none overflow-hidden opacity-5">
          <motion.div
            className="absolute inset-0 bg-gradient-to-b from-transparent via-white to-transparent h-1/3"
            animate={{
              y: ['-100%', '400%']
            }}
            transition={{
              duration: 8,
              repeat: Infinity,
              ease: "linear"
            }}
          />
        </div>
      )}

      {/* Custom glow effect */}
      {glowColor && (
        <div 
          className="absolute inset-0 rounded-lg opacity-20 blur-xl pointer-events-none"
          style={{ backgroundColor: glowColor }}
        />
      )}
    </>
  );

  if (interactive || onClick) {
    return (
      <motion.div
        initial={animate ? { opacity: 0, y: 20 } : false}
        animate={animate ? { opacity: 1, y: 0 } : false}
        whileHover={{ scale: 1.02, transition: { duration: 0.2 } }}
        whileTap={{ scale: 0.98 }}
        onClick={onClick}
        className={cn(
          "relative rounded-lg border overflow-hidden",
          variantStyles[variant],
          interactive && "cursor-pointer transition-all duration-200",
          interactive && "hover:border-neon-cyan/50 hover:shadow-neon-cyan/20",
          className
        )}
      >
        {cardContent}
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={animate ? { opacity: 0, y: 20 } : false}
      animate={animate ? { opacity: 1, y: 0 } : false}
      className={cn(
        "relative rounded-lg border overflow-hidden",
        variantStyles[variant],
        className
      )}
    >
      {cardContent}
    </motion.div>
  );
};

// Specialized card variants for common use cases

export const MetricCard: React.FC<{
  title: string;
  value: string | number;
  change?: number;
  changeLabel?: string;
  icon: LucideIcon;
  trend?: number;
  subtitle?: string;
  loading?: boolean;
  accentColor?: 'cyan' | 'green' | 'red' | 'purple' | 'yellow';
}> = ({ 
  title, 
  value, 
  change, 
  changeLabel,
  icon: Icon, 
  trend, 
  subtitle,
  loading = false,
  accentColor = 'cyan'
}) => {
  const colors = {
    cyan: 'text-neon-cyan border-neon-cyan/30 bg-neon-cyan/5',
    green: 'text-neon-green border-neon-green/30 bg-neon-green/5',
    red: 'text-neon-red border-neon-red/30 bg-neon-red/5',
    purple: 'text-neon-purple border-neon-purple/30 bg-neon-purple/5',
    yellow: 'text-neon-yellow border-neon-yellow/30 bg-neon-yellow/5'
  };

  return (
    <Card className="relative overflow-hidden">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-xs text-gray-500 uppercase tracking-wider">{title}</p>
          
          {loading ? (
            <div className="mt-2 space-y-2">
              <div className="h-8 w-24 bg-cyber-gray animate-pulse rounded" />
              <div className="h-3 w-16 bg-cyber-gray animate-pulse rounded" />
            </div>
          ) : (
            <>
              <p className="text-2xl font-bold text-white mt-1 font-mono">
                {value}
              </p>
              
              {(change !== undefined || subtitle) && (
                <div className="flex items-center mt-2 space-x-2">
                  {change !== undefined && (
                    <span className={cn(
                      "text-xs font-medium",
                      change >= 0 ? "text-neon-green" : "text-neon-red"
                    )}>
                      {change >= 0 ? '+' : ''}{change}%
                      {changeLabel && ` ${changeLabel}`}
                    </span>
                  )}
                  {subtitle && (
                    <span className="text-xs text-gray-500">{subtitle}</span>
                  )}
                </div>
              )}
            </>
          )}
        </div>
        
        <div className={cn(
          "p-3 rounded-lg",
          colors[trend !== undefined && trend < 0 ? 'red' : accentColor]
        )}>
          <Icon className="w-5 h-5" />
        </div>
      </div>
      
      {/* Trend indicator line */}
      {trend !== undefined && (
        <motion.div
          className={cn(
            "absolute bottom-0 left-0 h-0.5",
            trend >= 0 ? "bg-neon-green" : "bg-neon-red"
          )}
          initial={{ width: 0 }}
          animate={{ width: `${Math.abs(trend)}%` }}
          transition={{ duration: 1, ease: "easeOut" }}
        />
      )}
    </Card>
  );
};

// Empty state card - when there's no data to show
export const EmptyCard: React.FC<{
  icon?: LucideIcon;
  title: string;
  description?: string;
  action?: React.ReactNode;
}> = ({ icon: Icon, title, description, action }) => (
  <Card className="text-center py-12">
    {Icon && (
      <div className="mx-auto w-12 h-12 rounded-full bg-cyber-dark flex items-center justify-center mb-4">
        <Icon className="w-6 h-6 text-gray-500" />
      </div>
    )}
    <h3 className="text-lg font-medium text-white mb-2">{title}</h3>
    {description && (
      <p className="text-sm text-gray-500 mb-4 max-w-sm mx-auto">{description}</p>
    )}
    {action}
  </Card>
);
