import { pgTable, text, serial, integer, timestamp, boolean, varchar, date } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";
import { relations } from "drizzle-orm";

// Teacher table
export const teachers = pgTable("teachers", {
  id: serial("id").primaryKey(),
  username: varchar("username", { length: 50 }).notNull().unique(),
  password: varchar("password", { length: 100 }).notNull(),
  name: varchar("name", { length: 100 }).notNull(),
  subject: varchar("subject", { length: 100 }),
  email: varchar("email", { length: 100 }),
  createdAt: timestamp("created_at").defaultNow(),
});

export const teacherRelations = relations(teachers, ({ many }) => ({
  homeworks: many(homework),
  studentReports: many(studentReports),
}));

// Class table
export const classes = pgTable("classes", {
  id: serial("id").primaryKey(),
  name: varchar("name", { length: 10 }).notNull().unique(),
  displayName: varchar("display_name", { length: 50 }).notNull(),
  section: varchar("section", { length: 50 }).notNull(),
});

export const classRelations = relations(classes, ({ many }) => ({
  students: many(students),
  homeworks: many(homework),
}));

// Student table
export const students = pgTable("students", {
  id: serial("id").primaryKey(),
  name: varchar("name", { length: 100 }).notNull(),
  rollNumber: varchar("roll_number", { length: 20 }).notNull(),
  classId: integer("class_id").references(() => classes.id, { onDelete: "cascade" }),
});

export const studentRelations = relations(students, ({ one, many }) => ({
  class: one(classes, {
    fields: [students.classId],
    references: [classes.id],
  }),
  reports: many(studentReports),
}));

// Homework table
export const homework = pgTable("homework", {
  id: serial("id").primaryKey(),
  title: varchar("title", { length: 200 }).notNull(),
  description: text("description").notNull(),
  subject: varchar("subject", { length: 100 }).notNull(),
  classId: integer("class_id").references(() => classes.id, { onDelete: "cascade" }).notNull(),
  teacherId: integer("teacher_id").references(() => teachers.id, { onDelete: "cascade" }).notNull(),
  assignedDate: timestamp("assigned_date").defaultNow().notNull(),
  dueDate: date("due_date").notNull(),
  attachments: text("attachments"),
  status: varchar("status", { length: 20 }).default("active"),
});

export const homeworkRelations = relations(homework, ({ one }) => ({
  teacher: one(teachers, {
    fields: [homework.teacherId],
    references: [teachers.id],
  }),
  class: one(classes, {
    fields: [homework.classId],
    references: [classes.id],
  }),
}));

// Student Reports table
export const studentReports = pgTable("student_reports", {
  id: serial("id").primaryKey(),
  studentId: integer("student_id").references(() => students.id, { onDelete: "cascade" }).notNull(),
  teacherId: integer("teacher_id").references(() => teachers.id, { onDelete: "cascade" }).notNull(),
  content: text("content").notNull(),
  strengths: text("strengths"),
  areasToImprove: text("areas_to_improve"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const studentReportRelations = relations(studentReports, ({ one }) => ({
  student: one(students, {
    fields: [studentReports.studentId],
    references: [students.id],
  }),
  teacher: one(teachers, {
    fields: [studentReports.teacherId],
    references: [teachers.id],
  }),
}));

// Schemas for INSERT operations
export const insertTeacherSchema = createInsertSchema(teachers).omit({ id: true, createdAt: true });
export const insertClassSchema = createInsertSchema(classes).omit({ id: true });
export const insertStudentSchema = createInsertSchema(students).omit({ id: true });
export const insertHomeworkSchema = createInsertSchema(homework).omit({ id: true, assignedDate: true });
export const insertStudentReportSchema = createInsertSchema(studentReports).omit({ id: true, createdAt: true, updatedAt: true });

// Login schema
export const loginSchema = z.object({
  username: z.string().min(1, "Username is required"),
  password: z.string().min(1, "Password is required"),
});

// Types
export type InsertTeacher = z.infer<typeof insertTeacherSchema>;
export type Teacher = typeof teachers.$inferSelect;

export type InsertClass = z.infer<typeof insertClassSchema>;
export type Class = typeof classes.$inferSelect;

export type InsertStudent = z.infer<typeof insertStudentSchema>;
export type Student = typeof students.$inferSelect;

export type InsertHomework = z.infer<typeof insertHomeworkSchema>;
export type Homework = typeof homework.$inferSelect;

export type InsertStudentReport = z.infer<typeof insertStudentReportSchema>;
export type StudentReport = typeof studentReports.$inferSelect;

export type Login = z.infer<typeof loginSchema>;

Database .
import { Pool, neonConfig } from '@neondatabase/serverless';
import { drizzle } from 'drizzle-orm/neon-serverless';
import ws from "ws";
import * as schema from "@shared/schema";

neonConfig.webSocketConstructor = ws;

if (!process.env.DATABASE_URL) {
  throw new Error(
    "DATABASE_URL must be set. Did you forget to provision a database?",
  );
}

export const pool = new Pool({ connectionString: process.env.DATABASE_URL });
export const db = drizzle({ client: pool, schema });

import { Request, Response, NextFunction } from 'express';
import { z } from 'zod';
import { storage } from './storage';
import jwt from 'jsonwebtoken';
import { loginSchema } from '@shared/schema';

// Secret key for JWT
const JWT_SECRET = process.env.JWT_SECRET || 'sanskar-school-jwt-secret-key';

// Generate JWT token
export function generateToken(teacherId: number): string {
  return jwt.sign({ teacherId }, JWT_SECRET, { expiresIn: '24h' });
}

// Verify JWT token
export function verifyToken(token: string): { teacherId: number } | null {
  try {
    return jwt.verify(token, JWT_SECRET) as { teacherId: number };
  } catch (error) {
    return null;
  }
}

// Login middleware
export async function login(req: Request, res: Response) {
  try {
    // Validate request body
    const credentials = loginSchema.parse(req.body);
    
    // Check credentials
    const teacher = await storage.validateTeacherCredentials(
      credentials.username,
      credentials.password
    );
    
    if (!teacher) {
      return res.status(401).json({ message: 'Invalid username or password' });
    }
    
    // Generate JWT token
    const token = generateToken(teacher.id);
    
    // Return teacher data and token
    return res.status(200).json({
      token,
      teacher: {
        id: teacher.id,
        username: teacher.username,
        name: teacher.name,
        subject: teacher.subject,
        email: teacher.email
      }
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({ 
        message: 'Validation error', 
        errors: error.errors 
      });
    }
    
    return res.status(500).json({ message: 'Internal server error' });
  }
}

// Auth middleware to protect routes
export function authenticate(req: Request, res: Response, next: NextFunction) {
  // Get token from Authorization header
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ message: 'Authentication required' });
  }
  
  const token = authHeader.split(' ')[1];
  if (!token) {
    return res.status(401).json({ message: 'Authentication token required' });
  }
  
  // Verify token
  const payload = verifyToken(token);
  if (!payload) {
    return res.status(401).json({ message: 'Invalid or expired token' });
  }
  
  // Attach teacher ID to request
  req.body.teacherId = payload.teacherId;
  
  // Continue to the next middleware or route handler
  next();
}

import { eq, and, like, desc } from "drizzle-orm";
import { db } from "./db";
import { 
  teachers, 
  classes, 
  students, 
  homework, 
  studentReports, 
  type Teacher, 
  type InsertTeacher,
  type Class,
  type InsertClass,
  type Student,
  type InsertStudent,
  type Homework,
  type InsertHomework,
  type StudentReport,
  type InsertStudentReport
} from "@shared/schema";
import * as bcrypt from "bcrypt";

// Interface for storage operations
export interface IStorage {
  // Teacher operations
  getTeacher(id: number): Promise<Teacher | undefined>;
  getTeacherByUsername(username: string): Promise<Teacher | undefined>;
  createTeacher(teacher: InsertTeacher): Promise<Teacher>;
  validateTeacherCredentials(username: string, password: string): Promise<Teacher | null>;
  
  // Class operations
  getAllClasses(): Promise<Class[]>;
  getClassById(id: number): Promise<Class | undefined>;
  getClassByName(name: string): Promise<Class | undefined>;
  createClass(classData: InsertClass): Promise<Class>;
  
  // Student operations
  getStudentsInClass(classId: number): Promise<Student[]>;
  getStudentById(id: number): Promise<Student | undefined>;
  createStudent(student: InsertStudent): Promise<Student>;
  searchStudents(query: string, classId?: number): Promise<Student[]>;
  
  // Homework operations
  getHomeworkById(id: number): Promise<Homework | undefined>;
  getHomeworkForClass(classId: number): Promise<Homework[]>;
  getRecentHomework(classId: number, limit?: number): Promise<Homework[]>;
  createHomework(homework: InsertHomework): Promise<Homework>;
  updateHomework(id: number, data: Partial<InsertHomework>): Promise<Homework | undefined>;
  deleteHomework(id: number): Promise<boolean>;
  
  // Student Report operations
  getStudentReportById(id: number): Promise<StudentReport | undefined>;
  getStudentReportsForStudent(studentId: number): Promise<StudentReport[]>;
  getRecentStudentReports(classId: number, limit?: number): Promise<any[]>;
  createStudentReport(report: InsertStudentReport): Promise<StudentReport>;
  updateStudentReport(id: number, data: Partial<InsertStudentReport>): Promise<StudentReport | undefined>;
}

export class DatabaseStorage implements IStorage {
  // Teacher operations
  async getTeacher(id: number): Promise<Teacher | undefined> {
    const [teacher] = await db.select().from(teachers).where(eq(teachers.id, id));
    return teacher;
  }

  async getTeacherByUsername(username: string): Promise<Teacher | undefined> {
    const [teacher] = await db.select().from(teachers).where(eq(teachers.username, username));
    return teacher;
  }

  async createTeacher(teacherData: InsertTeacher): Promise<Teacher> {
    // Hash the password
    const hashedPassword = await bcrypt.hash(teacherData.password, 10);
    
    const [teacher] = await db
      .insert(teachers)
      .values({
        ...teacherData,
        password: hashedPassword
      })
      .returning();
    return teacher;
  }
  
  async validateTeacherCredentials(username: string, password: string): Promise<Teacher | null> {
    const teacher = await this.getTeacherByUsername(username);
    if (!teacher) return null;
    
    const isPasswordValid = await bcrypt.compare(password, teacher.password);
    if (!isPasswordValid) return null;
    
    return teacher;
  }

  // Class operations
  async getAllClasses(): Promise<Class[]> {
    return await db.select().from(classes);
  }

  async getClassById(id: number): Promise<Class | undefined> {
    const [classItem] = await db.select().from(classes).where(eq(classes.id, id));
    return classItem;
  }
  
  async getClassByName(name: string): Promise<Class | undefined> {
    const [classItem] = await db.select().from(classes).where(eq(classes.name, name));
    return classItem;
  }

  async createClass(classData: InsertClass): Promise<Class> {
    const [newClass] = await db
      .insert(classes)
      .values(classData)
      .returning();
    return newClass;
  }

  // Student operations
  async getStudentsInClass(classId: number): Promise<Student[]> {
    return await db
      .select()
      .from(students)
      .where(eq(students.classId, classId));
  }

  async getStudentById(id: number): Promise<Student | undefined> {
    const [student] = await db.select().from(students).where(eq(students.id, id));
    return student;
  }

  async createStudent(studentData: InsertStudent): Promise<Student> {
    const [student] = await db
      .insert(students)
      .values(studentData)
      .returning();
    return student;
  }
  
  async searchStudents(query: string, classId?: number): Promise<Student[]> {
    let filters = [like(students.name, `%${query}%`)];
    
    if (classId) {
      filters.push(eq(students.classId, classId));
    }
    
    return await db
      .select()
      .from(students)
      .where(and(...filters));
  }

  // Homework operations
  async getHomeworkById(id: number): Promise<Homework | undefined> {
    const [homeworkItem] = await db.select().from(homework).where(eq(homework.id, id));
    return homeworkItem;
  }

  async getHomeworkForClass(classId: number): Promise<Homework[]> {
    return await db
      .select()
      .from(homework)
      .where(eq(homework.classId, classId))
      .orderBy(desc(homework.assignedDate));
  }

  async getRecentHomework(classId: number, limit: number = 10): Promise<Homework[]> {
    return await db
      .select()
      .from(homework)
      .where(eq(homework.classId, classId))
      .orderBy(desc(homework.assignedDate))
      .limit(limit);
  }

  async createHomework(homeworkData: InsertHomework): Promise<Homework> {
    const [newHomework] = await db
      .insert(homework)
      .values(homeworkData)
      .returning();
    return newHomework;
  }

  async updateHomework(id: number, data: Partial<InsertHomework>): Promise<Homework | undefined> {
    const [updatedHomework] = await db
      .update(homework)
      .set(data)
      .where(eq(homework.id, id))
      .returning();
    return updatedHomework;
  }

  async deleteHomework(id: number): Promise<boolean> {
    const result = await db
      .delete(homework)
      .where(eq(homework.id, id))
      .returning();
    return result.length > 0;
  }

  // Student Report operations
  async getStudentReportById(id: number): Promise<StudentReport | undefined> {
    const [report] = await db.select().from(studentReports).where(eq(studentReports.id, id));
    return report;
  }

  async getStudentReportsForStudent(studentId: number): Promise<StudentReport[]> {
    return await db
      .select()
      .from(studentReports)
      .where(eq(studentReports.studentId, studentId))
      .orderBy(desc(studentReports.createdAt));
  }
  
  async getRecentStudentReports(classId: number, limit: number = 10): Promise<any[]> {
    // This requires joining tables to filter by class
    const classStudents = await this.getStudentsInClass(classId);
    const studentIds = classStudents.map(student => student.id);
    
    if (studentIds.length === 0) return [];
    
    const reports = await db
      .select({
        report: studentReports,
        student: students
      })
      .from(studentReports)
      .innerJoin(students, eq(studentReports.studentId, students.id))
      .where(
        studentIds.length === 1 
          ? eq(studentReports.studentId, studentIds[0])
          : studentReports.studentId.in(studentIds)
      )
      .orderBy(desc(studentReports.createdAt))
      .limit(limit);
      
    return reports;
  }

  async createStudentReport(reportData: InsertStudentReport): Promise<StudentReport> {
    const [report] = await db
      .insert(studentReports)
      .values(reportData)
      .returning();
    return report;
  }

  async updateStudentReport(id: number, data: Partial<InsertStudentReport>): Promise<StudentReport | undefined> {
    const [updatedReport] = await db
      .update(studentReports)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(studentReports.id, id))
      .returning();
    return updatedReport;
  }
}

export const storage = new DatabaseStorage();
import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { authenticate, login } from "./auth";

export async function registerRoutes(app: Express): Promise<Server> {
  // Auth routes
  app.post('/api/login', login);
  
  // Protected routes
  app.get('/api/teachers/me', authenticate, async (req, res) => {
    try {
      const teacherId = req.body.teacherId;
      const teacher = await storage.getTeacher(teacherId);
      
      if (!teacher) {
        return res.status(404).json({ message: 'Teacher not found' });
      }
      
      // Remove password from response
      const { password, ...teacherData } = teacher;
      res.json(teacherData);
    } catch (error) {
      res.status(500).json({ message: 'Failed to fetch teacher details' });
    }
  });
  
  // Classes routes
  app.get('/api/classes', authenticate, async (req, res) => {
    try {
      const classes = await storage.getAllClasses();
      res.json(classes);
    } catch (error) {
      res.status(500).json({ message: 'Failed to fetch classes' });
    }
  });
  
  app.get('/api/classes/:id', authenticate, async (req, res) => {
    try {
      const classId = parseInt(req.params.id);
      const classData = await storage.getClassById(classId);
      
      if (!classData) {
        return res.status(404).json({ message: 'Class not found' });
      }
      
      res.json(classData);
    } catch (error) {
      res.status(500).json({ message: 'Failed to fetch class details' });
    }
  });

  // Students routes  
  app.get('/api/classes/:classId/students', authenticate, async (req, res) => {
    try {
      const classId = parseInt(req.params.classId);
      const students = await storage.getStudentsInClass(classId);
      res.json(students);
    } catch (error) {
      res.status(500).json({ message: 'Failed to fetch students' });
    }
  });
  
  app.get('/api/students/search', authenticate, async (req, res) => {
    try {
      const query = req.query.q as string;
      const classId = req.query.classId ? parseInt(req.query.classId as string) : undefined;
      
      if (!query) {
        return res.status(400).json({ message: 'Search query is required' });
      }
      
      const students = await storage.searchStudents(query, classId);
      res.json(students);
    } catch (error) {
      res.status(500).json({ message: 'Failed to search students' });
    }
  });
  
  // Homework routes
  app.get('/api/classes/:classId/homework', authenticate, async (req, res) => {
    try {
      const classId = parseInt(req.params.classId);
      const homework = await storage.getHomeworkForClass(classId);
      res.json(homework);
    } catch (error) {
      res.status(500).json({ message: 'Failed to fetch homework' });
    }
  });
  
  app.get('/api/classes/:classId/homework/recent', authenticate, async (req, res) => {
    try {
      const classId = parseInt(req.params.classId);
      const limit = req.query.limit ? parseInt(req.query.limit as string) : 10;
      const homework = await storage.getRecentHomework(classId, limit);
      res.json(homework);
    } catch (error) {
      res.status(500).json({ message: 'Failed to fetch recent homework' });
    }
  });
  
  app.post('/api/homework', authenticate, async (req, res) => {
    try {
      const teacherId = req.body.teacherId;
      const homework = await storage.createHomework({
        ...req.body,
        teacherId
      });
      res.status(201).json(homework);
    } catch (error) {
      res.status(500).json({ message: 'Failed to create homework assignment' });
    }
  });
  
  app.put('/api/homework/:id', authenticate, async (req, res) => {
    try {
      const homeworkId = parseInt(req.params.id);
      const teacherId = req.body.teacherId;
      
      // Check if homework exists and belongs to the teacher
      const homework = await storage.getHomeworkById(homeworkId);
      
      if (!homework) {
        return res.status(404).json({ message: 'Homework not found' });
      }
      
      if (homework.teacherId !== teacherId) {
        return res.status(403).json({ message: 'Not authorized to update this homework' });
      }
      
      const updatedHomework = await storage.updateHomework(homeworkId, req.body);
      res.json(updatedHomework);
    } catch (error) {
      res.status(500).json({ message: 'Failed to update homework assignment' });
    }
  });
  
  app.delete('/api/homework/:id', authenticate, async (req, res) => {
    try {
      const homeworkId = parseInt(req.params.id);
      const teacherId = req.body.teacherId;
      
      // Check if homework exists and belongs to the teacher
      const homework = await storage.getHomeworkById(homeworkId);
      
      if (!homework) {
        return res.status(404).json({ message: 'Homework not found' });
      }
      
      if (homework.teacherId !== teacherId) {
        return res.status(403).json({ message: 'Not authorized to delete this homework' });
      }
      
      await storage.deleteHomework(homeworkId);
      res.status(204).send();
    } catch (error) {
      res.status(500).json({ message: 'Failed to delete homework assignment' });
    }
  });
  
  // Student Reports routes
  app.get('/api/students/:studentId/reports', authenticate, async (req, res) => {
    try {
      const studentId = parseInt(req.params.studentId);
      const reports = await storage.getStudentReportsForStudent(studentId);
      res.json(reports);
    } catch (error) {
      res.status(500).json({ message: 'Failed to fetch student reports' });
    }
  });
  
  app.get('/api/classes/:classId/reports/recent', authenticate, async (req, res) => {
    try {
      const classId = parseInt(req.params.classId);
      const limit = req.query.limit ? parseInt(req.query.limit as string) : 10;
      const reports = await storage.getRecentStudentReports(classId, limit);
      res.json(reports);
    } catch (error) {
      res.status(500).json({ message: 'Failed to fetch recent student reports' });
    }
  });
  
  app.post('/api/student-reports', authenticate, async (req, res) => {
    try {
      const teacherId = req.body.teacherId;
      const report = await storage.createStudentReport({
        ...req.body,
        teacherId
      });
      res.status(201).json(report);
    } catch (error) {
      res.status(500).json({ message: 'Failed to create student report' });
    }
  });
  
  app.put('/api/student-reports/:id', authenticate, async (req, res) => {
    try {
      const reportId = parseInt(req.params.id);
      const teacherId = req.body.teacherId;
      
      // Check if report exists and belongs to the teacher
      const report = await storage.getStudentReportById(reportId);
      
      if (!report) {
        return res.status(404).json({ message: 'Report not found' });
      }
      
      if (report.teacherId !== teacherId) {
        return res.status(403).json({ message: 'Not authorized to update this report' });
      }
      
      const updatedReport = await storage.updateStudentReport(reportId, req.body);
      res.json(updatedReport);
    } catch (error) {
      res.status(500).json({ message: 'Failed to update student report' });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}

import express, { type Request, Response, NextFunction } from "express";
import { registerRoutes } from "./routes";
import { setupVite, serveStatic, log } from "./vite";

const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

app.use((req, res, next) => {
  const start = Date.now();
  const path = req.path;
  let capturedJsonResponse: Record<string, any> | undefined = undefined;

  const originalResJson = res.json;
  res.json = function (bodyJson, ...args) {
    capturedJsonResponse = bodyJson;
    return originalResJson.apply(res, [bodyJson, ...args]);
  };

  res.on("finish", () => {
    const duration = Date.now() - start;
    if (path.startsWith("/api")) {
      let logLine = `${req.method} ${path} ${res.statusCode} in ${duration}ms`;
      if (capturedJsonResponse) {
        logLine += ` :: ${JSON.stringify(capturedJsonResponse)}`;
      }

      if (logLine.length > 80) {
        logLine = logLine.slice(0, 79) + "…";
      }

      log(logLine);
    }
  });

  next();
});

(async () => {
  const server = await registerRoutes(app);

  app.use((err: any, _req: Request, res: Response, _next: NextFunction) => {
    const status = err.status || err.statusCode || 500;
    const message = err.message || "Internal Server Error";

    res.status(status).json({ message });
    throw err;
  });

  // importantly only setup vite in development and after
  // setting up all the other routes so the catch-all route
  // doesn't interfere with the other routes
  if (app.get("env") === "development") {
    await setupVite(app, server);
  } else {
    serveStatic(app);
  }

  // ALWAYS serve the app on port 5000
  // this serves both the API and the client.
  // It is the only port that is not firewalled.
  const port = 5000;
  server.listen({
    port,
    host: "0.0.0.0",
    reusePort: true,
  }, () => {
    log(`serving on port ${port}`);
  });
})();
